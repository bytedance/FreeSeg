# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json
import logging
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

from .utils import load_binary_mask

logger = logging.getLogger(__name__)

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"id": 92, "name": "banner", "supercategory": "textile"},
    {"id": 93, "name": "blanket", "supercategory": "textile"},
    {"id": 94, "name": "branch", "supercategory": "plant"},
    {"id": 95, "name": "bridge", "supercategory": "building"},
    {"id": 96, "name": "building-other", "supercategory": "building"},
    {"id": 97, "name": "bush", "supercategory": "plant"},
    {"id": 98, "name": "cabinet", "supercategory": "furniture-stuff"},
    {"id": 99, "name": "cage", "supercategory": "structural"},
    {"id": 100, "name": "cardboard", "supercategory": "raw-material"},
    {"id": 101, "name": "carpet", "supercategory": "floor"},
    {"id": 102, "name": "ceiling-other", "supercategory": "ceiling"},
    {"id": 103, "name": "ceiling-tile", "supercategory": "ceiling"},
    {"id": 104, "name": "cloth", "supercategory": "textile"},
    {"id": 105, "name": "clothes", "supercategory": "textile"},
    {"id": 106, "name": "clouds", "supercategory": "sky"},
    {"id": 107, "name": "counter", "supercategory": "furniture-stuff"},
    {"id": 108, "name": "cupboard", "supercategory": "furniture-stuff"},
    {"id": 109, "name": "curtain", "supercategory": "textile"},
    {"id": 110, "name": "desk-stuff", "supercategory": "furniture-stuff"},
    {"id": 111, "name": "dirt", "supercategory": "ground"},
    {"id": 112, "name": "door-stuff", "supercategory": "furniture-stuff"},
    {"id": 113, "name": "fence", "supercategory": "structural"},
    {"id": 114, "name": "floor-marble", "supercategory": "floor"},
    {"id": 115, "name": "floor-other", "supercategory": "floor"},
    {"id": 116, "name": "floor-stone", "supercategory": "floor"},
    {"id": 117, "name": "floor-tile", "supercategory": "floor"},
    {"id": 118, "name": "floor-wood", "supercategory": "floor"},
    {"id": 119, "name": "flower", "supercategory": "plant"},
    {"id": 120, "name": "fog", "supercategory": "water"},
    {"id": 121, "name": "food-other", "supercategory": "food-stuff"},
    {"id": 122, "name": "fruit", "supercategory": "food-stuff"},
    {"id": 123, "name": "furniture-other", "supercategory": "furniture-stuff"},
    {"id": 124, "name": "grass", "supercategory": "plant"},
    {"id": 125, "name": "gravel", "supercategory": "ground"},
    {"id": 126, "name": "ground-other", "supercategory": "ground"},
    {"id": 127, "name": "hill", "supercategory": "solid"},
    {"id": 128, "name": "house", "supercategory": "building"},
    {"id": 129, "name": "leaves", "supercategory": "plant"},
    {"id": 130, "name": "light", "supercategory": "furniture-stuff"},
    {"id": 131, "name": "mat", "supercategory": "textile"},
    {"id": 132, "name": "metal", "supercategory": "raw-material"},
    {"id": 133, "name": "mirror-stuff", "supercategory": "furniture-stuff"},
    {"id": 134, "name": "moss", "supercategory": "plant"},
    {"id": 135, "name": "mountain", "supercategory": "solid"},
    {"id": 136, "name": "mud", "supercategory": "ground"},
    {"id": 137, "name": "napkin", "supercategory": "textile"},
    {"id": 138, "name": "net", "supercategory": "structural"},
    {"id": 139, "name": "paper", "supercategory": "raw-material"},
    {"id": 140, "name": "pavement", "supercategory": "ground"},
    {"id": 141, "name": "pillow", "supercategory": "textile"},
    {"id": 142, "name": "plant-other", "supercategory": "plant"},
    {"id": 143, "name": "plastic", "supercategory": "raw-material"},
    {"id": 144, "name": "platform", "supercategory": "ground"},
    {"id": 145, "name": "playingfield", "supercategory": "ground"},
    {"id": 146, "name": "railing", "supercategory": "structural"},
    {"id": 147, "name": "railroad", "supercategory": "ground"},
    {"id": 148, "name": "river", "supercategory": "water"},
    {"id": 149, "name": "road", "supercategory": "ground"},
    {"id": 150, "name": "rock", "supercategory": "solid"},
    {"id": 151, "name": "roof", "supercategory": "building"},
    {"id": 152, "name": "rug", "supercategory": "textile"},
    {"id": 153, "name": "salad", "supercategory": "food-stuff"},
    {"id": 154, "name": "sand", "supercategory": "ground"},
    {"id": 155, "name": "sea", "supercategory": "water"},
    {"id": 156, "name": "shelf", "supercategory": "furniture-stuff"},
    {"id": 157, "name": "sky-other", "supercategory": "sky"},
    {"id": 158, "name": "skyscraper", "supercategory": "building"},
    {"id": 159, "name": "snow", "supercategory": "ground"},
    {"id": 160, "name": "solid-other", "supercategory": "solid"},
    {"id": 161, "name": "stairs", "supercategory": "furniture-stuff"},
    {"id": 162, "name": "stone", "supercategory": "solid"},
    {"id": 163, "name": "straw", "supercategory": "plant"},
    {"id": 164, "name": "structural-other", "supercategory": "structural"},
    {"id": 165, "name": "table", "supercategory": "furniture-stuff"},
    {"id": 166, "name": "tent", "supercategory": "building"},
    {"id": 167, "name": "textile-other", "supercategory": "textile"},
    {"id": 168, "name": "towel", "supercategory": "textile"},
    {"id": 169, "name": "tree", "supercategory": "plant"},
    {"id": 170, "name": "vegetable", "supercategory": "food-stuff"},
    {"id": 171, "name": "wall-brick", "supercategory": "wall"},
    {"id": 172, "name": "wall-concrete", "supercategory": "wall"},
    {"id": 173, "name": "wall-other", "supercategory": "wall"},
    {"id": 174, "name": "wall-panel", "supercategory": "wall"},
    {"id": 175, "name": "wall-stone", "supercategory": "wall"},
    {"id": 176, "name": "wall-tile", "supercategory": "wall"},
    {"id": 177, "name": "wall-wood", "supercategory": "wall"},
    {"id": 178, "name": "water-other", "supercategory": "water"},
    {"id": 179, "name": "waterdrops", "supercategory": "water"},
    {"id": 180, "name": "window-blind", "supercategory": "window"},
    {"id": 181, "name": "window-other", "supercategory": "window"},
    {"id": 182, "name": "wood", "supercategory": "solid"},
]


COCO_BASE_CATEGORIES = [
    c
    for i, c in enumerate(COCO_CATEGORIES)
    if c["id"] - 1
    not in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
]
COCO_NOVEL_CATEGORIES = [
    c
    for i, c in enumerate(COCO_CATEGORIES)
    if c["id"] - 1
    in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
]


def _get_coco_stuff_meta(cat_list):
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    thing_ids = [k["id"] for k in cat_list if "isthing" in k]
    stuff_ids = [k["id"] for k in cat_list]

    
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    
    thing_classes = [k["name"] for k in cat_list if "isthing" in k]
    stuff_classes = [k["name"] for k in cat_list]

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def load_full_binary_mask(json_file, pan_dir, gt_root, image_root, meta, gt_ext="png", image_ext="jpg"):
    """

    `label_count_file` contains a dictionary like:
    ```
    """
    label_count_file = gt_root + "_label_count.json"
    with open(label_count_file) as f:
        label_count_dict = json.load(f)

    data_dicts = load_sem_seg(gt_root, image_root, gt_ext, image_ext)
    flattened_data_dicts = []
    for data in data_dicts:

        category_per_image = label_count_dict[
            os.path.basename(data["sem_seg_file_name"])
        ]
        data["task"] = "sem_seg"
        flattened_data = [
            dict(**{"category_id": cat}, **data) for cat in category_per_image
        ]
        
        flattened_data_dicts.extend(flattened_data)
    
    pan_data_dicts = load_coco_full_json(json_file, image_root, pan_dir, gt_root, meta)
    
    # instance
    for data in pan_data_dicts:
        
        for seg_info in data["segments_info"]:
            if seg_info["iscrowd"] == 0 and seg_info["isthing"] == True:
                flattened_data = dict(**seg_info, **{
                    "task": "ins_seg",
                    "file_name": data["file_name"],
                    "pan_seg_file_name": data["pan_seg_file_name"],
                    "sem_seg_file_name": data["sem_seg_file_name"]
                })
                flattened_data_dicts.append(flattened_data)

    # panoptic    
    for data in pan_data_dicts:
        
        for seg_info in data["segments_info"]:
            if seg_info["iscrowd"] == 0:
                flattened_data = dict(**seg_info, **{
                    "task": "pan_seg",
                    "file_name": data["file_name"],
                    "pan_seg_file_name": data["pan_seg_file_name"],
                    "sem_seg_file_name": data["sem_seg_file_name"]
                })
                flattened_data_dicts.append(flattened_data)
    
    logger.info(
        "Loaded {} images with flattened semantic segmentation from {}".format(
            len(flattened_data_dicts), image_root
        )
    )
    return flattened_data_dicts

def load_coco_full_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    with PathManager.open(json_file) as f:
        json_info = json.load(f)
    

    input_files = sorted(
        (os.path.join(image_dir, f) for f in PathManager.ls(image_dir) if f.endswith("jpg")),
        key=lambda file_path: file2id(image_dir, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_dir, f) for f in PathManager.ls(gt_dir) if f.endswith("png")),
        key=lambda file_path: file2id(gt_dir, file_path),
    )

    semseg_files = sorted(
        (os.path.join(semseg_dir, f) for f in PathManager.ls(semseg_dir) if f.endswith("png")),
        key=lambda file_path: file2id(semseg_dir, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_dir)

    # Use the intersection
    if len(input_files) != len(gt_files) or len(input_files) != len(semseg_files):
        logger.warn(
            "Directory {}, {}, and {} has {}, {}, and {} files, respectively.".format(
                image_dir, gt_dir, semseg_dir, len(input_files), len(gt_files), len(semseg_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len("jpg")] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len("png")] for f in gt_files]
        semseg_basenames = [os.path.basename(f)[: -len("png")] for f in semseg_files]
        intersect = list(set(input_basenames) & set(gt_basenames) & set(semseg_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_dir, f + "jpg") for f in intersect]
        gt_files = [os.path.join(gt_dir, f + "png") for f in intersect]
        semseg_files = [os.path.join(semseg_dir, f + "png") for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_dir)
    )

    annotations_dicts = {anno["file_name"]: anno for anno in json_info["annotations"]}
    dataset_dicts = []
    for (img_path, gt_path, semseg_path) in zip(input_files, gt_files, semseg_files):
        record = {}
        record["file_name"] = img_path
        record["pan_seg_file_name"] = gt_path
        record["sem_seg_file_name"] = semseg_path
        
        basename = os.path.basename(gt_path)
        annotation = annotations_dicts[basename]
        image_id = int(annotation["image_id"])
        record["image_id"] = image_id
        segments_info = [_convert_category_id(x, meta) for x in annotation["segments_info"] 
                    if x["category_id"] in meta["stuff_dataset_id_to_contiguous_id"]]
        record["segments_info"] = segments_info
        dataset_dicts.append(record)
        

    
    assert len(dataset_dicts), f"No images found in {image_dir}!"
    assert PathManager.isfile(dataset_dicts[0]["file_name"]), dataset_dicts[0]["file_name"]
    assert PathManager.isfile(dataset_dicts[0]["pan_seg_file_name"]), dataset_dicts[0]["pan_seg_file_name"]
    assert PathManager.isfile(dataset_dicts[0]["sem_seg_file_name"]), dataset_dicts[0]["sem_seg_file_name"]
    return dataset_dicts


def register_all_coco_full_164k(root):
    root = os.path.join(root, "coco")
    meta = _get_coco_stuff_meta(COCO_CATEGORIES)
    base_meta = _get_coco_stuff_meta(COCO_BASE_CATEGORIES)
    novel_meta = _get_coco_stuff_meta(COCO_NOVEL_CATEGORIES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "train2017", "stuffthingmaps_panoptic_detectron2/train2017"),
        ("test", "val2017", "stuffthingmaps_panoptic_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"coco_2017_{name}_full_task"
        
        if name == "test":
            prefix_instances = "coco_2017_" + "val"
            panoptic_json = os.path.join(root, "panoptic_detectron2/panoptic_val2017.json") 
            panoptic_root = os.path.join(root, "panoptic_detectron2/val2017") 
        else:
            prefix_instances = "coco_2017_" + name
            panoptic_json = os.path.join(root, "panoptic_detectron2/panoptic_train2017.json") 
            panoptic_root = os.path.join(root, "panoptic_detectron2/train2017") 
        instances_meta = MetadataCatalog.get(prefix_instances)
        instances_json = instances_meta.json_file
        
        
        DatasetCatalog.register(
            all_name,
            lambda panoptic_json=panoptic_json, 
                    image_dir=image_dir, 
                    panoptic_root=panoptic_root, 
                    gt_dir=gt_dir: load_coco_full_json(
                panoptic_json, 
                image_dir, 
                panoptic_root, 
                gt_dir, 
                meta
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            json_file=instances_json,
            panoptic_json=panoptic_json,
            panoptic_root=panoptic_root,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
                "novel_thing": [
                    meta["stuff_classes"].index(n)
                    for i, n in enumerate(novel_meta["stuff_classes"])
                    if COCO_NOVEL_CATEGORIES[i].get("isthing", 0) == 1
                ],
                "novel_stuff": [
                    meta["stuff_classes"].index(n)
                    for i, n in enumerate(novel_meta["stuff_classes"])
                    if COCO_NOVEL_CATEGORIES[i].get("isthing", 0) == 0
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )
        # classification
        DatasetCatalog.register(
            all_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )

        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_base")
        base_name = f"coco_2017_{name}_full_task_base"
        
        panoptic_json_base = panoptic_json[:-5] + "_base.json"
        panoptic_root_base = panoptic_root + "_base"
        DatasetCatalog.register(
            base_name,  
            lambda panoptic_json=panoptic_json_base, 
                    image_dir=image_dir, 
                    panoptic_root=panoptic_root_base, 
                    gt_dir=gt_dir: load_coco_full_json(
                panoptic_json, 
                image_dir, 
                panoptic_root, 
                gt_dir, 
                base_meta
            ),
        )
        MetadataCatalog.get(base_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            json_file=instances_json,
            panoptic_json=panoptic_json_base,
            panoptic_root=panoptic_root_base,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            **base_meta,
        )
        # classification
        DatasetCatalog.register(
            base_name + "_classification",
            lambda panoptic_json=panoptic_json_base, 
                    image_dir=image_dir, 
                    panoptic_root=panoptic_root_base, 
                    gt_dir=gt_dir: load_full_binary_mask(
                panoptic_json, 
                panoptic_root, 
                gt_dir, 
                image_dir, 
                base_meta, 
                gt_ext="png", 
                image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            **base_meta,
        )
        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_novel")
        novel_name = f"coco_2017_{name}_full_task_novel"
        panoptic_json_novel = panoptic_json[:-5] + "_novel.json"
        panoptic_root_novel = panoptic_root + "_novel"

        DatasetCatalog.register(
            novel_name,
            lambda panoptic_json=panoptic_json_novel, 
                    image_dir=image_dir, 
                    panoptic_root=panoptic_root_novel, 
                    gt_dir=gt_dir: load_coco_full_json(
                panoptic_json, 
                image_dir, 
                panoptic_root, 
                gt_dir, 
                novel_meta
            ),
        )
        MetadataCatalog.get(novel_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            json_file=instances_json,
            panoptic_json=panoptic_json_novel,
            panoptic_root=panoptic_root_novel,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            **novel_meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_full_164k(_root)
