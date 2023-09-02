import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import jsonlines

class FashionBase(Dataset):
    def __init__(self,
                 jsonl_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = jsonl_file
        self.data_root = data_root
        self.image_paths = []
        self.text_list = []
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                self.image_paths.append(obj['image_path'])
                self.text_list.append(obj['text'])
        self._length = len(self.image_paths)
        self.labels = {
            "image_path": [os.path.join(self.data_root, l)
                               for l in self.image_paths],
            "caption":[l for l in self.text_list],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["image_path"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    
class FashionTrain(FashionBase):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file="/root/autodl-tmp/tv_train.jsonl", data_root="/root/autodl-tmp/train", **kwargs)

class FashionValid(FashionBase):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file="/root/autodl-tmp/tv_valid.jsonl", data_root="/root/autodl-tmp/valid", **kwargs)
