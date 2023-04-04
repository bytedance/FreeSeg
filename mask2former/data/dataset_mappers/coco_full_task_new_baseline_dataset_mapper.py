# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["COCOFullTaskNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOFullTaskNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticSemNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            augs = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
            dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "tfm_gens": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if is_train else -1,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )
        else:
            sem_seg_gt = None
        
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
            
            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id
            
            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
            
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
            
            dataset_dict["pan_instances"] = instances

        
        ###### for instance segmentation
            ins_instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"] and segment_info["isthing"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
            
            classes = np.array(classes)
            ins_instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                ins_instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                ins_instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                ins_instances.gt_masks = masks.tensor
                ins_instances.gt_boxes = masks.get_bounding_boxes()
            
            dataset_dict["ins_instances"] = ins_instances

        # semantic segmentation
        if sem_seg_gt is not None:
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                image = F.pad(image, padding_size, value=128).contiguous()
                
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w
            dataset_dict["image"] = image

            dataset_dict["sem_seg"] = sem_seg_gt.long()


            # Prepare per-category binary masks
            sem_seg_gt = sem_seg_gt.numpy()
            sem_seg_instances = Instances(image_shape)
            sem_classes = np.unique(sem_seg_gt)
            # remove ignored region
            sem_classes = sem_classes[sem_classes != self.ignore_label]
            sem_seg_instances.gt_classes = torch.tensor(sem_classes, dtype=torch.int64)

            sem_masks = []
            for class_id in sem_classes:
                sem_masks.append(sem_seg_gt == class_id)

            if len(sem_masks) == 0:
                # Some image does not have annotation (all ignored)
                sem_seg_instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )
            else:
                sem_masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in sem_masks
                        ]
                    )
                )
                sem_seg_instances.gt_masks = sem_masks.tensor
            
            dataset_dict["sem_instances"] = sem_seg_instances

        return dataset_dict
