import argparse
import os
import os.path as osp
import json
import copy
import shutil
from functools import partial
from glob import glob

from panopticapi.utils import rgb2id, id2rgb

import mmcv
import numpy as np
from PIL import Image

import torch
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

COCO_LEN = 123287

full_clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    26: 24,
    27: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    45: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    66: 60,
    69: 61,
    71: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    83: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    91: 80,
    92: 81,
    93: 82,
    94: 83,
    95: 84,
    96: 85,
    97: 86,
    98: 87,
    99: 88,
    100: 89,
    101: 90,
    102: 91,
    103: 92,
    104: 93,
    105: 94,
    106: 95,
    107: 96,
    108: 97,
    109: 98,
    110: 99,
    111: 100,
    112: 101,
    113: 102,
    114: 103,
    115: 104,
    116: 105,
    117: 106,
    118: 107,
    119: 108,
    120: 109,
    121: 110,
    122: 111,
    123: 112,
    124: 113,
    125: 114,
    126: 115,
    127: 116,
    128: 117,
    129: 118,
    130: 119,
    131: 120,
    132: 121,
    133: 122,
    134: 123,
    135: 124,
    136: 125,
    137: 126,
    138: 127,
    139: 128,
    140: 129,
    141: 130,
    142: 131,
    143: 132,
    144: 133,
    145: 134,
    146: 135,
    147: 136,
    148: 137,
    149: 138,
    150: 139,
    151: 140,
    152: 141,
    153: 142,
    154: 143,
    155: 144,
    156: 145,
    157: 146,
    158: 147,
    159: 148,
    160: 149,
    161: 150,
    162: 151,
    163: 152,
    164: 153,
    165: 154,
    166: 155,
    167: 156,
    168: 157,
    169: 158,
    170: 159,
    171: 160,
    172: 161,
    173: 162,
    174: 163,
    175: 164,
    176: 165,
    177: 166,
    178: 167,
    179: 168,
    180: 169,
    181: 170,
    255: 255,
}

panoptic_clsID_to_trID = {
    183: 171,
    184: 172,
    185: 173,
    186: 174,
    187: 175,
    188: 176,
    189: 177,
    190: 178,
    191: 179,
    192: 180,
    193: 181,
    194: 182,
    195: 183,
    196: 184,
    197: 185,
    198: 186,
    199: 187,
}

novel_clsID = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
pan_clsID = [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}
pan_clsID_to_trID = {k: i for i, k in enumerate(pan_clsID)}



def convert_to_trainID(
    maskpath, out_mask_dir, panoptic_root, out_panoptic_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    # add panoptic cls to stuff
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    if is_train:
        obj_annos = train_obj_annos_copy
    else:
        obj_annos = val_obj_annos_copy
    file_name = osp.basename(maskpath)
    obj_anno = obj_annos[file_name]
    segments = obj_anno["segments_info"]
    panoptic = osp.join(panoptic_root, file_name)
    panoptic_mask = np.asarray(Image.open(panoptic), dtype=np.uint32)
    panoptic_mask = rgb2id(panoptic_mask)
    panoptic_copy = np.zeros_like(panoptic_mask, dtype=np.uint32) 

    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID


    seg_filename = (
        osp.join(out_mask_dir, "train2017" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val2017" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")

    mask_array = np.asarray(Image.open(maskpath).convert("RGB"), dtype=np.uint32)
    mask_id = rgb2id(mask_array)
    mask_one = np.ones_like(mask, dtype=np.uint8)
    categories = [seg["category_id"] for seg in segments]
    new_segments = []
    for seg in segments:
        cat_id = seg["category_id"]
        id_cls = seg["id"]
        if cat_id-1 in clsID_to_trID.keys():
            panoptic_copy[panoptic_mask == id_cls] = id_cls
            new_segments.append(seg)
            

    for clsID, trID in clsID_to_trID.items():
        if clsID > 90 and clsID < 255 and clsID+1 not in categories:
            id = np.unique(mask_id[mask == clsID])
            panoptic_copy[mask == clsID] = id
            area = mask_one[mask == clsID].sum()
            if area > 0:
                mask_box = np.zeros_like(mask, dtype=np.uint8)
                mask_box[mask == clsID] = 1
                mask_box = torch.Tensor(mask_box).unsqueeze(0)
                bbox = BitMasks(mask_box > 0).get_bounding_boxes()
                out_box = [int(idx) for idx in bbox.tensor[0]]
                
                segment_info = {'id': int(id[0]),
                                'category_id': int(clsID+1),
                                'iscrowd': 0,
                                'bbox': out_box,
                                'area': int(area)}
                new_segments.append(segment_info)
    
    if is_train:
        train_obj_annos_copy[file_name]["segments_info"] = new_segments
    else:
        val_obj_annos_copy[file_name]["segments_info"] = new_segments

    panoptic_filename = (
        osp.join(out_panoptic_mask_dir, "train2017" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_panoptic_mask_dir, "val2017" + suffix, osp.basename(maskpath))
    )
    
    panoptic_copy = id2rgb(panoptic_copy)
    Image.fromarray(panoptic_copy).save(panoptic_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("coco_path", help="coco path")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    panoptic_train_json_path = osp.join(coco_path, "annotations", "panoptic_train2017.json")
    panoptic_val_json_path = osp.join(coco_path, "annotations", "panoptic_val2017.json")
    
    panoptic_train_root = osp.join(coco_path, "panoptic_train2017")
    panoptic_val_root = osp.join(coco_path, "panoptic_val2017")
    nproc = args.nproc
    print(full_clsID_to_trID)
    print(base_clsID_to_trID)
    print(novel_clsID_to_trID)
    out_dir = args.out_dir or coco_path
    out_mask_dir = osp.join(out_dir, "stuffthingmaps_panoptic_detectron2")
    out_panoptic_mask_dir = osp.join(out_dir, "panoptic_detectron2")

    
    for dir_name in [
        "train2017",
        "val2017",
        "train2017_base",
        "train2017_novel",
        "val2017_base",
        "val2017_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
        os.makedirs(osp.join(out_panoptic_mask_dir, dir_name), exist_ok=True)
    
    train_list = glob(osp.join(coco_path, "stuffthingmaps", "train2017", "*.png"))
    test_list = glob(osp.join(coco_path, "stuffthingmaps", "val2017", "*.png"))
    
    assert (
        len(train_list) + len(test_list)
    ) == COCO_LEN, "Wrong length of list {} & {}".format(
        len(train_list), len(test_list)
    )


    with open(panoptic_train_json_path) as f:
        train_obj = json.load(f)
    train_obj_annos = {anno["file_name"]: anno for anno in train_obj["annotations"]}
    train_obj_all = copy.deepcopy(train_obj)

    with open(panoptic_val_json_path) as f:
        val_obj = json.load(f)
    val_obj_annos = {anno["file_name"]: anno for anno in val_obj["annotations"]}
    val_obj_all = copy.deepcopy(val_obj)
    
    global train_obj_annos_copy
    global val_obj_annos_copy
    
    if args.nproc > 1:
        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir=out_panoptic_mask_dir,
                is_train=True,
            ),
            train_list,
            nproc=nproc,
        )
        suffix = ""
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir, 
                panoptic_root=panoptic_val_root, 
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=False,
            ),
            test_list,
            nproc=nproc,
        )
        suffix = ""
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
            nproc=nproc,
        )
        suffix = "_base"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_val_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
            nproc=nproc,
        )
        suffix = "_base"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
            nproc=nproc,
        )
        suffix = "_novel"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_val_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
            nproc=nproc,
        )
        suffix = "_novel"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

    else:
        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID, 
                out_mask_dir=out_mask_dir, 
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir, 
                is_train=True),
            train_list,
        )
        suffix = ""
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir, 
                panoptic_root=panoptic_val_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir, 
                is_train=False),
            test_list,
        )
        suffix = ""
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
        )
        suffix = "_base"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_val_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
        )
        suffix = "_base"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

        train_obj_annos_copy = copy.deepcopy(train_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_train_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
        )
        suffix = "_novel"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_train2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            train_obj_all["annotations"] = list(train_obj_annos_copy.values())
            json.dump(train_obj_all, f)

        val_obj_annos_copy = copy.deepcopy(val_obj_annos)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                panoptic_root=panoptic_val_root,
                out_panoptic_mask_dir = out_panoptic_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
        )
        suffix = "_novel"
        panoptic_json_filename = osp.join(out_panoptic_mask_dir, "panoptic_val2017" + suffix + ".json")
        with open(panoptic_json_filename, "w") as f:
            val_obj_all["annotations"] = list(val_obj_annos_copy.values())
            json.dump(val_obj_all, f)

    print("Done!")


if __name__ == "__main__":
    main()
