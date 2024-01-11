#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

ALL_ID = [i for i in range(150)]
NOVEL_ID = [9, 15, 30, 37, 49, 60, 74, 81, 89, 99, 112, 128, 136, 143, 149]
BASE_ID = [i for i in ALL_ID if i not in NOVEL_ID]

def convert(input, output, index=None):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    if index is not None:
        mapping = {i: k for k, i in enumerate(index)}
        img = np.vectorize(lambda x: mapping[x] if x in mapping else 255)(
            img.astype(np.float)
        ).astype(np.uint8)
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = (
        Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
    )
    for name in ["training", "validation"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)

        base_name = name + "_base"
        output_dir = dataset_dir / "annotations_detectron2" / base_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file, BASE_ID)
        
        novel_name = name + "_novel"
        output_dir = dataset_dir / "annotations_detectron2" / novel_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file, NOVEL_ID)
