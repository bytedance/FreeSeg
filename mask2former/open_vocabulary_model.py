# Copyright (c) Facebook, Inc. and its affiliates.
from cgitb import text
import logging
import copy
import random
import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.transforms import ResizeTransform
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer


@META_ARCH_REGISTRY.register()
class OpenVocabulary(MaskFormer):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        task_names: list,
        criterion: nn.Module,
        num_queries: int,
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            semantic_on=semantic_on,
            instance_on=instance_on,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
       
        self._region_clip_adapter = region_clip_adapter

        self.task_names = task_names
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

        self.test_topk_per_image = test_topk_per_image


    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER, cfg.INPUT.TASK_NAME)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER, cfg.INPUT.TASK_NAME
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=0.4,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )
        
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["task_names"] = cfg.INPUT.TASK_NAME
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT
        init_kwargs["test_topk_per_image"] = cfg.TEST.DETECTIONS_PER_IMAGE
        init_kwargs["metadata"] = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        init_kwargs["semantic_on"] = "semantic segmentation." in cfg.INPUT.TASK_NAME
        init_kwargs["instance_on"] = "instance segmentation." in cfg.INPUT.TASK_NAME
        init_kwargs["panoptic_on"] = "panoptic segmentation." in cfg.INPUT.TASK_NAME

        init_kwargs["cfg"] = cfg

        return init_kwargs


    def forward(self, batched_inputs, text_labels=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if text_labels == None:
            dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
            assert len(set(dataset_name)) == 1
            dataset_name = dataset_name[0]
        else:
            dataset_name = " " 
        
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        features = self.backbone(images.tensor)
        
        if text_labels == None:
            class_names = self.get_class_name_list(dataset_name)
        else: 
            class_names = text_labels

        if self.training:
            task_name = random.choice(self.task_names)
        
            text_features = self.clip_adapter.get_text_features(class_names, task_name)

            outputs, fused_text_features = self.sem_seg_head(features, text_features)

            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
            )

            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if task_name == "semantic segmentation.":
                gt_instances = [x["sem_instances"].to(self.device) for x in batched_inputs]
            elif task_name == "instance segmentation.":
                gt_instances = [x["ins_instances"].to(self.device) for x in batched_inputs]
            elif task_name == "panoptic segmentation.":
                gt_instances = [x["pan_instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            task_name = "semantic segmentation."
        
            text_features = self.clip_adapter.get_text_features(class_names, task_name)

            outputs, fused_text_features = self.sem_seg_head(features, text_features)

            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
            )

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )
            
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

            # instance segmentation inference
            if self.instance_on:
                
                task_name = "instance segmentation."
                
                text_features = self.clip_adapter.get_text_features(class_names, task_name)

                outputs, fused_text_features = self.sem_seg_head(features, text_features)

                outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                    text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
                )

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )

                for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                )):
                    height = image_size[0]
                    width = image_size[1]
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )
                    image = input_per_image["image"].to(self.device)

                    instance_r = self.instance_inference(
                        mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                    )
                
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    # process results
                    if instance_r.pred_masks.shape[0] > 0:
                        cur_device = instance_r.pred_masks.device
                        instance_mask = instance_r.pred_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        ori_h, ori_w, num_mask = instance_mask.shape[0], instance_mask.shape[1], instance_mask.shape[2]
                        transform = ResizeTransform(ori_h, ori_w, height, width)
                        
                        if num_mask > 3:
                            instance_mask_list = [transform.apply_segmentation(instance_mask[:, :, p1-3:p1]) for p1 in range(3, num_mask+1, 3)]
                            if np.mod(num_mask, 3) > 0:
                                mask_last = transform.apply_segmentation(instance_mask[:, :, -np.mod(num_mask, 3):])
                                instance_mask_list.append(mask_last)
                            instance_mask = np.concatenate(instance_mask_list, axis=2)
                        else:
                            instance_mask = transform.apply_segmentation(instance_mask)
                        
                        instance_mask = torch.tensor(instance_mask).permute(2, 0, 1).to(cur_device)
                        instance_r.pred_masks = instance_mask

                        if not instance_r.pred_boxes is None:
                            instance_boxes = instance_r.pred_boxes.tensor
                            x1_coords, x2_coords = instance_boxes[:, :2], instance_boxes[:, 2:]
                            x1_coords = transform.apply_coords(x1_coords)
                            x2_coords = transform.apply_coords(x2_coords)
                            instance_boxes = torch.cat((x1_coords, x2_coords), dim=1)
                            instance_r.pred_boxes = Boxes(instance_boxes)
                    
                    processed_results[i]["instances"] = instance_r
            
            # panoptic segmentation inference
            if self.panoptic_on:

                task_name = "panoptic segmentation."
                
                text_features = self.clip_adapter.get_text_features(class_names, task_name)

                outputs, fused_text_features = self.sem_seg_head(features, text_features)

                outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                    text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
                )

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )

                for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                )):
                    height = image_size[0]
                    width = image_size[1]
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )
                    image = input_per_image["image"].to(self.device)

                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                    )
                
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    # process results
                    cur_device = panoptic_r[0].device
                    panoptic_mask = panoptic_r[0].cpu().numpy().astype(np.uint8)
                    ori_h, ori_w = panoptic_mask.shape[0], panoptic_mask.shape[1]
                    transform = ResizeTransform(ori_h, ori_w, height, width)
                    panoptic_mask = transform.apply_segmentation(panoptic_mask)
                    panoptic_r[0] = torch.tensor(panoptic_mask).to(cur_device)

                    segment_info = panoptic_r[1]
                    cur_seg_ids = list(torch.unique(panoptic_r[0]))
                    segment_info = [seg_info for seg_info in segment_info if seg_info["id"] in cur_seg_ids]
                    panoptic_r[1] = segment_info
                    processed_results[i]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        
        # get the classification result from clip model
        
        if self.clip_ensemble:
            
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred, normalize=True
            )
            
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        
        
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]

        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred).argmax(0)

        scores, labels = F.softmax(mask_cls / 0.01, dim=-1).max(-1)
        keep = labels.ne(self.sem_seg_head.num_classes) 
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return [panoptic_seg, segments_info]
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                pred_class_name = class_names[pred_class]
                isthing = pred_class_name in self.metadata.thing_classes 

                
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_masks[k] >= 0.5) & (sem_maps == pred_class)
                mask_area = mask.sum().item()

                if original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    if isthing and cur_scores[k] < 0.5:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    
            panoptic_res = [panoptic_seg, segments_info]
            return panoptic_res

    def instance_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        
        image_size = mask_pred.shape[-2:]
        num_classes = mask_cls.shape[-1]
        
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred.sigmoid(), normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
           
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        
        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred.sigmoid()).argmax(0)


        scores = F.softmax(mask_cls / 0.01, dim=-1)[:, :-1]
        scores_per_image, labels_per_image = scores.max(-1)


        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                pred_class_name = class_names[lab]
                keep[i] = pred_class_name in self.metadata.thing_classes 

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        class_mask_memory = {}
        keep = torch.zeros_like(scores_per_image).bool()
    
        for k in range(labels_per_image.shape[0]):
            
            pred_class = labels_per_image[k]
            original_area = (mask_pred[k] >= 0.5).sum().item()
            
            mask = (mask_pred[k] >= 0.5) & (sem_maps == pred_class)
            mask_area = mask.sum().item()

            if mask_area > 0 and original_area > 0 and scores_per_image[k] > 0.5:
                if mask_area / original_area > self.overlap_threshold:
                    keep[k] = True

                    if lab in class_mask_memory.keys():
                        class_mask_memory[lab].append(k)
                    else: 
                        class_mask_memory[lab] = [k]
        
        for cls_id, idx_list in class_mask_memory.items():
            mask_area_list = [(mask_pred[i] >= 0.5).sum().item() for i in idx_list]
            max_area = np.max(np.array(mask_area_list))
            max_idx = np.argmax(np.array(mask_area_list))
            union_mask = torch.zeros_like(mask_pred[0]).bool()
            for i, idx in enumerate(idx_list):
                if i != max_idx:
                    union_mask = (union_mask ==True) | (mask_pred[idx] >= 0.5) 
            union_mask_area = union_mask.sum().item()
            if union_mask_area / max_area > 0.8:
                keep[idx_list[max_idx]] = False

        
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]
        
        result = Instances(image_size)
        
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names


    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter
