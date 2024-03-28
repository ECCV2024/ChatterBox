import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import torchvision


from .conversation import get_default_conv_template
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
import re

import utils.transforms as T
from .box_ops import box_xyxy_to_cxcywh, box_xywh_to_cxcywh
from PIL import Image
import random

DEFAULT_REGION_TOKEN = "<region>"

img_size = 512

expanded_names = {
    'person': 1, 'human': 1, 'man': 1, 'woman': 1, 'girl': 1, 'boy': 1,
    'bicycle': 2, 'bike': 2,
    'car': 3,
    'motorcycle': 4,
    'airplane': 5, 'plane': 5, 'aeroplane': 5, 'jet': 5,
    'bus': 6, 'shuttle': 6, 'trolley': 6, 'motorbus': 6, 'coach': 6,
    'train': 7, 'metro': 7, 'subway': 7,
    'truck': 8, 'lorry': 8, 'van': 8, 'rig': 8,
    'boat': 9, 'ship': 9, 'vessel': 9, 'yacht': 9,
    'traffic light': 10, 'stoplight': 10, 'semaphore': 10, 'signal light': 10, 'traffic signal': 10,
    'fire hydrant': 11, 'hydrant': 11, 'fireplug': 11, 'water plug': 11, 'fire pump': 11,
    'stop sign': 13, 'stop signal': 13,
    'parking meter': 14,
    'bench': 15, 'seat': 15, 'pew': 15, 'stool': 15, 'ottoman': 15,
    'bird': 16, 'avian': 16, 'fowl': 16, 'raptor': 16,
    'cat': 17, 'kitty': 17, 'kitten': 17,
    'dog': 18, 'canine': 18, 'pooch': 18, 'hound': 18, 'pup': 18,
    'horse': 19, 'stallion': 19, 'pony': 19, 'steed': 19,
    'sheep': 20, 'lamb': 20, 'flock': 20, 'goat': 20,
    'cow': 21, 'cattle': 21, 'calf': 21,
    'elephant': 22,
    'bear': 23, 'bruin': 23, 'cub': 23,
    'zebra': 24,
    'giraffe': 25,
    'backpack': 27, 'knapsack': 27, 'rucksack': 27, 'knapsack': 27,
    'umbrella': 28, 'parasol': 28, 'brolly': 28, 'sunshade': 28,
    'handbag': 31, 'bag': 31, 'purse': 31,
    'tie': 32, 'necktie': 32, 'cravat': 32,
    'suitcase': 33, 'luggage': 33, 'baggage': 33, 'trunk': 33,
    'frisbee': 34,
    'skis': 35, 'ski': 35,
    'snowboard': 36,
    'sports ball': 37, 'ball': 37,
    'kite': 38,
    'baseball bat': 39, 'bat': 39, 'club': 39,
    'baseball glove': 40, 'glove': 40, 'baseball mitt': 40,
    'skateboard': 41, 'deck': 41, 'cruiser': 41,
    'surfboard': 42, 'surfing board': 42, 'surfing plank': 42, 'shortboard': 42, 'longboard': 42,
    'tennis racket': 43, 'tennis racquet': 43, 'racket': 43,
    'bottle': 44, 'container': 44, 'vessel': 44, 'jar': 44, 'canteen': 44,
    'wine glass': 46, 'wine goblet': 46, 'wine tumbler': 46,
    'cup': 47, 'glass': 47, 'mug': 47,
    'fork': 48,
    'knife': 49, 'sword': 49, 'dagger': 49, 'knives': 49,
    'spoon': 50, 'scoop': 50,
    'bowl': 51,
    'banana': 52,
    'apple': 53,
    'sandwich': 54,
    'orange': 55,
    'broccoli': 56, 'cabbage': 56, 'cauliflower': 56,
    'carrot': 57,
    'hot dog': 58,
    'pizza': 59,
    'donut': 60, 'doughnut': 60,
    'cake': 61,
    'chair': 62,
    'couch': 63, 'sofa': 63, 'settee': 63, 'lounge': 63,
    'potted plant': 64, 'houseplant': 64, 'indoor plant': 64,
    'bed': 65,
    'dining table': 67, 'table': 67, 'desk': 67,
    'toilet': 70, 'restroom': 70, 'lavatory': 70,
    'tv': 72, 'television': 72,
    'laptop': 73, 'computer': 73, 'ultrabook': 73, 'netbook': 73,
    'mouse': 74,
    'remote': 75, 'controller': 75, 'zapper': 75,
    'keyboard': 76, 'keypad': 76, 'keyset': 76,
    'cell phone': 77, 'phone': 77, 'mobile phone': 77, 'smartphone': 77,
    'microwave': 78,
    'oven': 79,
    'toaster': 80,
    'sink': 81,
    'refrigerator': 82, 'fridge': 82,
    'book': 84,
    'clock': 85,
    'vase': 86,
    'scissors': 87, 'shears': 87, 'snips': 87,
    'teddy bear': 88, 'teddy': 88,
    'hair drier': 89,
    'toothbrush': 90
}


class RefCOCOGroundingDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor_aux = CLIPImageProcessor.from_pretrained("../CLIP/clip-vit-large-patch14")

        self.data_path = os.path.join(base_root)
        with open(os.path.join('./', 'jack_refcoco_refcoco+_grounding_v30.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding
        # """Normalize pixel values and pad to a square input."""
        # # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # # # Pad
        # # h, w = x.shape[-2:]
        # # padh = self.img_size - h
        # # padw = self.img_size - w
        # # x = F.pad(x, (0, padw, 0, padh))

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):
        # print('bboxes_raw  >> ', bboxes_raw)

        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xywh_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        # boxes = box_xyxy_to_cxcywh(bboxes_raw)
        # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
                '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in \
                bboxes_str[0] \
                or 'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
            return []
        # print('bboxes_str[0]  >>> ', bboxes_str)
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this

            label = [self.jack_json[idx]["category_id"]]

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    if "<" in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    sentence['value'] = sentence['value'] + '[VG]'

                    bboxes_human.append([])

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:

                    if len(bboxes_human) > 0:
                        bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                        gt_bboxes = self.strbbox2bbox(bboxes_str)
                        if len(gt_bboxes) > 0:
                            bboxes_gpt.append(gt_bboxes)

                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if sum([len(bboxes) for bboxes in bboxes_gpt]) > 0 or len(source) == j + 1:
                        sentence["value"] = sentence["value"]  # + ' [VG] '

                conv.append_message(role, sentence["value"])

                if len(bboxes_human) > 0 and j % 2 == 1:
                    break

            conversations.append(conv.get_prompt())


            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            # images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction

            # regions = self.extract_regions(bboxes_human, torch.from_numpy(images_ori))  # cast to llm together with image
            if conversations[0].count("<im_start>") == 1 and conversations[0].count("[VG]") == 1:
                # print('len of bboxes > 0  ... ', bboxes_human)
                bbox = bboxes_human[0]
                break

        return (
            images,
            images_clip,
            conversations,
            bboxes_gpt,
            label,
        )


class COCOGroundingDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor_aux = CLIPImageProcessor.from_pretrained("../CLIP/clip-vit-large-patch14")

        self.data_path = os.path.join(base_root)
        # with open(os.path.join('./', 'jack_v20_ground.json')) as f:
        with open(os.path.join('./', 'jack_v30_ground_coco.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding
        # """Normalize pixel values and pad to a square input."""
        # # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # # # Pad
        # # h, w = x.shape[-2:]
        # # padh = self.img_size - h
        # # padw = self.img_size - w
        # # x = F.pad(x, (0, padw, 0, padh))

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):
        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        # boxes = box_xyxy_to_cxcywh(bboxes_raw)
        # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
                '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in \
                bboxes_str[0] \
                or 'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
            return []
        # print('bboxes_str[0]  >>> ', bboxes_str)
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this

            label = [self.jack_json[idx]["category_id"]]

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    if "<" in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    sentence['value'] = sentence['value'] + '[VG]'

                    bboxes_human.append([])

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:

                    if len(bboxes_human) > 0:
                        bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                        gt_bboxes = self.strbbox2bbox(bboxes_str)
                        if len(gt_bboxes) > 0:
                            bboxes_gpt.append(gt_bboxes)

                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if sum([len(bboxes) for bboxes in bboxes_gpt]) > 0 or len(source) == j + 1:
                        sentence["value"] = sentence["value"]  # + ' [VG] '

                conv.append_message(role, sentence["value"])

                if len(bboxes_human) > 0 and j % 2 == 1:
                    break

            conversations.append(conv.get_prompt())


            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            # images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction
            # print('COCOGroundingDataset >>', bboxes_gpt)

            # regions = self.extract_regions(bboxes_human, torch.from_numpy(images_ori))  # cast to llm together with image
            if conversations[0].count("<im_start>") == 1 and conversations[0].count("[VG]") == 1:
                # print('len of bboxes > 0  ... ', bboxes_human)
                bbox = bboxes_human[0]
                break


        return (
            images,
            images_clip,
            conversations,
            bboxes_gpt,
            label,
        )


class CBMRGDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor_aux = CLIPImageProcessor.from_pretrained("../CLIP/clip-vit-large-patch14")

        self.data_path = os.path.join(base_root)
        with open(os.path.join('./', 'CB_GND.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

        self.replace_names = ['the region', 'this region']

        self.first_q = "This is an image. Can you answer the next questions about the specific regions in the image?  "
        self.first_a = "Sure, I will answer your questions.  "

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding
        # """Normalize pixel values and pad to a square input."""
        # # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # # # Pad
        # # h, w = x.shape[-2:]
        # # padh = self.img_size - h
        # # padw = self.img_size - w
        # # x = F.pad(x, (0, padw, 0, padh))

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):
        # print('bboxes_raw  >> ', bboxes_raw)

        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        # boxes = box_xyxy_to_cxcywh(bboxes_raw)
        # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
                '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in \
                bboxes_str[0] \
                or 'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
            return []
        # print('bboxes_str[0]  >>> ', bboxes_str)
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            replace_name = -2
            # random sample convs from start_id -> note logical reasoning NOT has this
            len_conv = len(source)
            start_id = 0
            if len_conv > 2:
                rand_id = random.randint(0, len_conv - 1)
                start_id = \
                random.sample([rand_id, int(len_conv // 2), int(len_conv // 4), int(len_conv // 6), int(len_conv // 8)],
                              1)[0]
                start_id = start_id // 2 * 2
                source = source[start_id:]

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            label = -1

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    # bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    bboxes_human.append([])

                    ########################## find coco classes #########################
                    sentence_next = source[j + 1]
                    bboxes_str_next = re.findall(r"<(.+?)>", sentence_next["value"])
                    bboxes_next = self.strbbox2bbox(bboxes_str_next)
                    if len(bboxes_next) == 1:
                        ins_name = bboxes_str_next[0].split('<')[-1].split(':')[0]

                        coco_expand_keys = expanded_names.keys()

                        for key in coco_expand_keys:
                            if key in ins_name:
                                label = expanded_names[key]
                                break
                    ######################################################################
                    if label != -1:
                        sentence["value"] = sentence["value"] + '[VG]'

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:

                    if label != -1:
                        bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                        gt_bboxes = self.strbbox2bbox(bboxes_str)
                        if len(gt_bboxes) > 0:
                            bboxes_gpt.append(gt_bboxes)

                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if replace_name == j - 1:
                        if ins_name in sentence["value"]:
                            sentence["value"] = sentence["value"].replace(ins_name, name_re)

                conv.append_message(role, sentence["value"])

                if label != -1 and j % 2 == 1:
                    break

            conversations.append(conv.get_prompt())


            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            # images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction
            # print('JackGroundingDataset >>', bboxes_gpt)

            # regions = self.extract_regions(bboxes_human, torch.from_numpy(images_ori))  # cast to llm together with image
            # if len(bboxes_human) > 0 and conversations[0].count("<im_start>") == 1 and conversations[0].count("[VG]") == 1:
            if conversations[0].count("<im_start>") == 1 and conversations[0].count("[VG]") == 1 and label != -1:
                break


        return (
            images,
            images_clip,
            conversations,
            bboxes_gpt,
            [label],
        )


class CBLCDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor_aux = CLIPImageProcessor.from_pretrained("../CLIP/clip-vit-large-patch14")

        self.data_path = os.path.join(base_root)
        with open(os.path.join('./', 'jack_logic_v30.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding
        # """Normalize pixel values and pad to a square input."""
        # # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # # # Pad
        # # h, w = x.shape[-2:]
        # # padh = self.img_size - h
        # # padw = self.img_size - w
        # # x = F.pad(x, (0, padw, 0, padh))

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):

        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        # boxes = box_xyxy_to_cxcywh(bboxes_raw)
        # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0:
            return []
        # print('bboxes_str[0]  >>> ', bboxes_str)
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            # print('images_clip  >>> ', images_clip.shape)
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this
            # resize = images.shape[:2]

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            label = -1

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    if "<" in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    sentence['value'] = sentence['value'].replace('[it]', 'it')

                    ########################## find coco classes #########################
                    sentence_next = source[j + 1]
                    bboxes_str_next = re.findall(r"<(.+?)>", sentence_next["value"])
                    bboxes_next = self.strbbox2bbox(bboxes_str_next)
                    # print('bboxes_str_next   >>>', bboxes_str_next)
                    if len(bboxes_next) == 1:
                        ins_name = bboxes_str_next[0].split('<')[-1].split(':')[0]

                        # print('ins_name   >>>', ins_name)

                        coco_expand_keys = expanded_names.keys()

                        for key in coco_expand_keys:
                            if key in ins_name:
                                # print('key   >>>', key)
                                label = expanded_names[key]
                                break
                    ######################################################################
                    if label != -1:
                        sentence["value"] = sentence["value"] + '[VG]'

                    bboxes_human.append([])

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:
                    if label != -1:
                        bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                        gt_bboxes = self.strbbox2bbox(bboxes_str)
                        if len(gt_bboxes) == 1:
                            bboxes_gpt.append(gt_bboxes)

                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                conv.append_message(role, sentence["value"])

                if label != -1 and j % 2 == 1:
                    break

            conversations.append(conv.get_prompt())

            # print('JackLogicGroundingDataset >>', conversations, len(conversations))

            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            # images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction
            # print('JackLogicGroundingDataset >>', bboxes_gpt)

            # regions = self.extract_regions(bboxes_human, torch.from_numpy(images_ori))  # cast to llm together with image
            if conversations[0].count("<im_start>") == 1 and conversations[0].count("[VG]") == 1 and label != -1:
                break

        # print(conversations, len(conversations), bbox)

        return (
            images,
            images_clip,
            conversations,
            bboxes_gpt,
            [label],
        )


def collate_fn(batch, tokenizer=None):
    images_list = []
    images_clip_list = []
    conversation_list = []
    bboxes_gt_list = []
    label_list = []
    offset_list = [0]
    cnt = 0
    for (
            images,
            images_clip,
            conversations,
            bbox,
            label,
    ) in batch:
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        bboxes_gt_list.append(bbox)
        cnt += len(conversations)
        offset_list.append(cnt)

    tokenize_data = tokenizer(
        conversation_list,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = tokenize_data.input_ids
    attention_masks = tokenize_data.attention_mask

    IGNORE_TOKEN_ID = -100
    conv = get_default_conv_template("vicuna").copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        # print('len of rounds  >>> ', rounds, sep, conv.sep2, len(rounds))
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "offset": torch.LongTensor(offset_list),
        "labels": targets,
        "label_list": label_list,
        "bboxes_gt_list": bboxes_gt_list,
    }


class GroundingDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 768

    # ignore_label = 255

    def __init__(
            self,
            base_image_dir,
            base_coco_dir,
            tokenizer,
            vision_tower,
    ):
        dataset = "refcocoground||cocoground||jackground||jacklogicground"
        # sample_rate = [3, 3, 3, 1]
        sample_rate = [2, 3, 2, 2]

        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "refcocoground":
                self.all_datasets.append(
                    RefCOCOGroundingDataset(
                        base_root='../MSCOCO2014/images/train2014/',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                    )
                )
            elif dataset == "cocoground":
                self.all_datasets.append(
                    COCOGroundingDataset(
                        base_root='../MSCOCO2017/train2017',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                    )
                )
            elif dataset == "jackground":
                self.all_datasets.append(
                    CBMRGDataset(
                        base_root='../VG/VG/',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                    )
                )
            elif dataset == "jacklogicground":
                self.all_datasets.append(
                    CBLCDataset(
                        base_root='../VG/VG/',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                    )
                )

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]

        # data = self.all_datasets[0]
        inference = False
        # print('data[0]', data[0])
        return data[0]
