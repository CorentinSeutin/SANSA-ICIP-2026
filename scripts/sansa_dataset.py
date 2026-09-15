import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import ANSWER_LIST, LONG_QUESTION_LIST, SHORT_QUESTION_LIST


class SANSADataset(torch.utils.data.Dataset):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    image_extensions = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        precision="fp32",
        image_size=224,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        dataset_dir = Path(base_image_dir).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"[INFO][ERROR] SANSA dataset directory not found at: {dataset_dir}")

        image_paths = sorted(
            path for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_extensions
        )

        self.samples = []
        for img_path in image_paths:
            annotation_path = img_path.with_suffix(".json")
            self.samples.append((img_path, annotation_path))

        print(f"[INFO] Number of SANSA samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def preprocess(self, image):
        height, width = image.shape[-2:]
        image = (image - self.pixel_mean) / self.pixel_std
        return F.pad(image, (0, self.img_size - width, 0, self.img_size - height))

    def __getitem__(self, idx):
        img_path, annotation_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        mask, prompts, is_sentence = get_mask_from_json(str(annotation_path), image)

        masks = torch.from_numpy((mask == 1).astype(np.float32)).unsqueeze(0)
        prompt = prompts[0]

        if is_sentence:
            question = random.choice(LONG_QUESTION_LIST).format(sent=prompt)
        else:
            question = random.choice(SHORT_QUESTION_LIST).format(class_name=prompt.lower())
        questions = [question]

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], random.choice(ANSWER_LIST))
        conversations = [conv.get_prompt()]

        resized_image = self.transform.apply_image(image)
        resize = resized_image.shape[:2]
        sam_image = self.preprocess(
            torch.from_numpy(resized_image).permute(2, 0, 1).contiguous()
        )

        label = torch.full(
            masks.shape[-2:], 
            self.ignore_label, 
            dtype=torch.long
        )

        return (
            str(img_path), sam_image, image_clip, conversations, masks, label,
            resize, questions, prompts, False,
        )
