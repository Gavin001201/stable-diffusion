import os
import json
import albumentations
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


class CocoBase(Dataset):
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, 
                 size=None, 
                 dataroot="", 
                 datajson="", 
                 interpolation="bicubic",
                 flip_p=0.5):
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        with open(datajson) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))

    def __len__(self):
        return len(self.labels["image_ids"])

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        image = Image.open(image_path)(img_path)

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
        image = (image / 127.5 - 1.0).astype(np.float32)
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   "caption": str(caption[0]),
                   "img_path": img_path,
                    }
        return example

class CocoImagesAndCaptionsTrain(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="/mnt/workspace/Project/taming-transformers2.0/data/coco/train2017",
                         datajson="/mnt/workspace/Project/taming-transformers2.0/data/coco/annotations/captions_train2017.json",
                         **kwargs)

class CocoImagesAndCaptionsValidation(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="/mnt/workspace/Project/taming-transformers2.0/data/coco/val2017",
                         datajson="/mnt/workspace/Project/taming-transformers2.0/data/coco/annotations/captions_val2017.json",
                         **kwargs)