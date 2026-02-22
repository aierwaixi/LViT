# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from bert_embedding import BertEmbedding
import re

def _pad_or_truncate_text(text_arr, max_len):
    # Keep token length fixed so DataLoader can stack tensors in a batch.
    if text_arr.ndim == 1:
        text_arr = np.expand_dims(text_arr, axis=0)
    cur = text_arr.shape[0]
    dim = text_arr.shape[1]
    if cur >= max_len:
        return text_arr[:max_len, :]
    pad = np.zeros((max_len - cur, dim), dtype=text_arr.dtype)
    return np.concatenate([text_arr, pad], axis=0)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1], dtype=np.float32)
        text = _pad_or_truncate_text(text, 14)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224,
                 allowed_names=None) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path, self.output_path = self._resolve_io_paths(dataset_path)
        self.images_list = [x for x in os.listdir(self.input_path) if self._is_image_file(x)]
        self.mask_list = [x for x in os.listdir(self.output_path) if self._is_image_file(x)]
        if allowed_names is not None:
            self.images_list = [x for x in self.images_list if self._name_in_allowed(x, allowed_names)]
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def _is_image_file(name):
        return os.path.splitext(name.lower())[1] in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    @staticmethod
    def _numeric_signature(name):
        stem = os.path.splitext(os.path.basename(name))[0]
        nums = re.findall(r"\d+", stem)
        if not nums:
            return ""
        return "-".join(str(int(x)) for x in nums)

    @staticmethod
    def _normalized_stem(name):
        stem = os.path.splitext(os.path.basename(name))[0].lower()
        for p in ["mask_", "mask-", "img_", "img-", "image_", "image-"]:
            if stem.startswith(p):
                stem = stem[len(p):]
        stem = stem.replace("_mask", "").replace("-mask", "")
        return stem

    def _name_in_allowed(self, image_name, allowed):
        stem = os.path.splitext(image_name)[0]
        sig = self._numeric_signature(image_name)
        num_tokens = [str(int(x)) for x in re.findall(r"\d+", stem)]
        compact_name = re.sub(r"[^a-z0-9]", "", image_name.lower())
        compact_stem = re.sub(r"[^a-z0-9]", "", stem.lower())
        compact_norm = re.sub(r"[^a-z0-9]", "", self._normalized_stem(image_name))
        return (
            image_name in allowed
            or stem in allowed
            or self._normalized_stem(image_name) in allowed
            or compact_name in allowed
            or compact_stem in allowed
            or compact_norm in allowed
            or (sig in allowed and sig != "")
            or any(t in allowed for t in num_tokens)
        )

    def _resolve_io_paths(self, dataset_path):
        cands = [
            (os.path.join(dataset_path, "img"), os.path.join(dataset_path, "labelcol")),
            (os.path.join(dataset_path, "images"), os.path.join(dataset_path, "masks")),
            (os.path.join(dataset_path, "frames"), os.path.join(dataset_path, "masks")),
        ]
        for inp, out in cands:
            if os.path.isdir(inp) and os.path.isdir(out):
                return inp, out
        raise FileNotFoundError("Cannot find image/mask folders under {}".format(dataset_path))

    def _find_mask_filename(self, image_filename):
        stem = os.path.splitext(image_filename)[0]
        candidates = [
            stem + ".png",
            stem + ".jpg",
            stem + ".jpeg",
            "mask_" + stem + ".png",
            "mask_" + stem + ".jpg",
            "mask-" + stem + ".png",
            "mask-" + stem + ".jpg",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(self.output_path, c)):
                return c
        # fallback by normalized stem / numeric id
        norm = self._normalized_stem(image_filename)
        num = self._numeric_signature(image_filename)
        for m in self.mask_list:
            if self._normalized_stem(m) == norm:
                return m
            if num and self._numeric_signature(m) == num:
                return m
        raise FileNotFoundError("Cannot find mask for image {}".format(image_filename))

    def _lookup_text(self, image_filename, mask_filename):
        keys = [
            image_filename,
            os.path.splitext(image_filename)[0],
            mask_filename,
            os.path.splitext(mask_filename)[0],
            self._normalized_stem(image_filename),
            self._numeric_signature(image_filename),
        ]
        for k in keys:
            if k in self.rowtext:
                return self.rowtext[k]
        return "infected lung region"

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        mask_filename = self._find_mask_filename(image_filename)
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self._lookup_text(image_filename, mask_filename)
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1], dtype=np.float32)
        text = _pad_or_truncate_text(text, 10)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
