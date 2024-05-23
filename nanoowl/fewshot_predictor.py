# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Union

import PIL.Image
import torch
from torch import nn

from .image_preprocessor import ImagePreprocessor
from .owl_predictor import (
    OwlDecodeOutput,
    OwlEncodeImageOutput,
    OwlEncodeTextOutput,
    OwlPredictor,
) 
__all__ = [
    "FewshotPredictor",
]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.activate0 = nn.GELU()
        for i in range(len(hidden_dims)-1):
            setattr(self, f'fc{i+1}', nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            setattr(self, f'activate{i+1}', nn.GELU())
        self.fcout = nn.Linear(hidden_dims[-1], output_dim)
        self.final_activate = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = self.activate0(x)
        for i in range(len(self.hidden_dims)-1):
            x = getattr(self, f'fc{i+1}')(x)
            x = getattr(self, f'activate{i+1}')(x)
        x = self.fcout(x)
        x = self.final_activate(x)
        return x


clf = MLP(768, [1024, 2048, 1024, 512, 256, 128], 2)
clf.load_state_dict(torch.load('model/mlp_total_data_cross_val_2_best.pth'))
clf.eval()

class FewshotPredictor(torch.nn.Module):
    def __init__(
        self,
        owl_predictor: Optional[OwlPredictor] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.owl_predictor = OwlPredictor() if owl_predictor is None else owl_predictor
        self.image_preprocessor = (
            ImagePreprocessor().to(device).eval()
            if image_preprocessor is None
            else image_preprocessor
        )

    @torch.no_grad()
    def predict(
        self,
        image: PIL.Image,
        query_embeddings: List,
        threshold: Union[int, float, List[Union[int, float]]] = 0.1,
        pad_square: bool = True,
    ) -> OwlDecodeOutput:
        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )

        image_encodings = self.owl_predictor.encode_rois(
            image_tensor, rois, pad_square=pad_square
        )

        return self.decode(image_encodings, query_embeddings, threshold)

    def decode(
        self,
        image_output: OwlEncodeImageOutput,
        query_embeds,
        threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    ) -> OwlDecodeOutput:
        num_input_images = image_output.image_class_embeds.shape[0]
        print(f"{num_input_images=}")

        image_class_embeds = image_output.image_class_embeds
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )

        if isinstance(threshold, (int, float)):
            threshold = [threshold] * len(
                query_embeds
            )  # apply single threshold to all labels

        query_embeds = torch.concat(query_embeds, dim=0)
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale

        scores_sigmoid = torch.sigmoid(logits)
        scores_max = scores_sigmoid.max(dim=-1)
        labels = scores_max.indices
        scores = scores_max.values
        masks = []
        for i, thresh in enumerate(threshold):
            label_mask = labels == i
            score_mask = scores > thresh
            obj_mask = torch.logical_and(label_mask, score_mask)
            masks.append(obj_mask)
        mask = masks[0]
        for mask_t in masks[1:]:
            mask = torch.logical_or(mask, mask_t)

        input_indices = torch.arange(
            0, num_input_images, dtype=labels.dtype, device=labels.device
        )
        input_indices = input_indices[:, None].repeat(1, self.owl_predictor.num_patches)

        return OwlDecodeOutput(
            labels=labels[mask],
            scores=scores[mask],
            boxes=image_output.pred_boxes[mask],
            input_indices=input_indices[mask],
        )

    def encode_query_image(
        self,
        image: PIL.Image,
        text: str,
        pad_square: bool = True,
    ) -> torch.Tensor:
        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        text_encodings = self.encode_text([text])

        rois = torch.tensor(
            [[0, 0, image.width, image.height]],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )

        image_encodings = self.owl_predictor.encode_rois(
            image_tensor, rois, pad_square=pad_square
        )
        return self.find_best_encoding(image_encodings, text_encodings)

    def encode_text(self, text) -> OwlEncodeTextOutput:
        return self.owl_predictor.encode_text(text)

    @staticmethod
    def find_best_encoding(
        image_output: OwlEncodeImageOutput,
        text_output: OwlEncodeTextOutput,
    ) -> torch.Tensor:
        image_class_embeds = image_output.image_class_embeds
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = text_output.text_embeds
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale

        scores_sigmoid = torch.sigmoid(logits)
        scores_max = scores_sigmoid.max(dim=-1)
        scores = scores_max.values
        best = torch.argmax(scores).item()
        best_embed = image_class_embeds[:, best]
        return best_embed