# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import io
import os
from pathlib import Path
from typing import Callable, Optional

import h5py
from PIL import Image
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
        h5_path: str,
        transform: Optional[Callable] = None,
    ):
        """H5 Dataset.
        The dataset assumes that data is organized as:
            "class_name"
                "img_name"
                "img_name"
                "img_name"
            "class_name"
                "img_name"
                "img_name"
                "img_name"

        Args:
            dataset (str): dataset name.
            h5_path (str): path of the h5 file.
            transform (Callable): pipeline of transformations. Defaults to None.
            pre_parsed_paths_file Optional[str]: path of the pre-parsed paths files.
                This allows the user to specify the file names and their classes in this format:
                {class}/{file} CLASS-ID
                {class}/{file} CLASS-ID
                {class}/{file} CLASS-ID
                If this is None, this object will automatically find all the files,
                but might take a while if the dataset is large. Defaults to None.
        """

        self.h5_path = h5_path
        self.h5_file = None
        self.transform = transform

        assert dataset in ["imagenet100", "imagenet"]

        self._load_h5_data_info()

        # filter if needed to avoid having a copy of imagenet100 data
        if dataset == "imagenet100":
            script_folder = Path(os.path.dirname(__file__))
            classes_file = script_folder / "dataset_subset" / "imagenet100_classes.txt"
            with open(classes_file, "r") as f:
                self.classes = f.readline().strip().split()
            self.classes = sorted(self.classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            self._data = list(filter(lambda entry: entry[0] in self.classes, self._data))

    def _load_h5_data_info(self):
        self._data = []
        h5_data_info_file = os.path.splitext(self.h5_path)[0] + ".txt"
        if not os.path.isfile(h5_data_info_file):
            temp_h5_file = h5py.File(self.h5_path, "r")

            # collect data from the h5 file directly
            self.classes, self.class_to_idx = self._find_classes(self.h5_file)
            for class_name in self.classes:
                y = self.class_to_idx[class_name]
                for img_name in temp_h5_file[class_name].keys():
                    self._data.append((class_name, img_name, y))

            # save the info locally to speed up sequential executions
            with open(h5_data_info_file, "w") as f:
                for class_name, img_name, y in self._data:
                    f.write(f"{class_name}/{img_name} {y}\n")
        else:
            # load data info file that was already generated by previous runs
            with open(h5_data_info_file, "r") as f:
                for line in f:
                    class_name_img, y = line.strip().split(" ")
                    class_name, img_name = class_name_img.split("/")
                    self._data.append((class_name, img_name, y))

    def _find_classes(self, h5_file: h5py.File):
        classes = sorted(h5_file.keys())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_img(self, class_name: str, img: str):
        img = self.h5_file[class_name][img][:]
        img = Image.open(io.BytesIO(img)).convert("RGB")
        return img

    def __getitem__(self, index: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        class_name, img, y = self._data[index]

        x = self._load_img(class_name, img)
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self._data)
