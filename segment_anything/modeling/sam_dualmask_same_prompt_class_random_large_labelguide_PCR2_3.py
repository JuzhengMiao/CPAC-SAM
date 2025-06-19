# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_prompt_large import MaskDecoder_prompt_large
from .prompt_encoder_prompt_class import PromptEncoder_prompt_class
import numpy as np

from skimage.measure import label
import cv2
from PIL import Image, ImageDraw
import math

def MaskToBoxSimple(mask):
    mask = mask.squeeze()
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()

    return [x0,y0,x1,y1]

class Sam_dualmask_same_prompt_class_random_large_labelguide_PCR2_3(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder_prompt_class,
        mask_decoder1: MaskDecoder_prompt_large,
        mask_decoder2: MaskDecoder_prompt_large,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder1 = mask_decoder1
        self.mask_decoder2 = mask_decoder2
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size, prompt_idx = -1, prompt_mode = None, labels = None, scaling = 8):
        # prompt_idx indicates which branch is used to generate prompts
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size, prompt_idx, prompt_mode, labels, scaling) 
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size, prompt_idx, prompt, labels, scaling):  
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)
        embed_new = image_embeddings.clone()

        if prompt_idx == -1:
            scaling1, scaling2 = scaling, scaling
        else:
            scaling1, scaling2 = scaling[0], scaling[1]


        if prompt_idx == 0: 
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
        
            low_res_masks1, iou_predictions1, embed = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            # generate prompts based on the coarse prediction
            points_prompt, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks1, image_size, embed_new, labels, scaling1)  
            points_prompt2, box_prompt2, mask_prompt2 = self.prompt_generate_random_fast(low_res_masks1, image_size, embed_new, labels, scaling2)  

            if prompt == 'point':
                sparse_embeddings = []
                dense_embeddings = []
                for i in range(input_images.shape[0]):
                    prompt_temp = (points_prompt[0][i].unsqueeze(0), points_prompt[1][i].unsqueeze(0))
                    sparse_embeddings_i, dense_embeddings_i = self.prompt_encoder(
                        points=prompt_temp, boxes=None, masks=None
                    )
                    sparse_embeddings.append(sparse_embeddings_i)
                    dense_embeddings.append(dense_embeddings_i)
                
                sparse_embeddings2 = []
                dense_embeddings2 = []
                for i in range(input_images.shape[0]):
                    prompt_temp2 = (points_prompt2[0][i].unsqueeze(0), points_prompt2[1][i].unsqueeze(0))
                    sparse_embeddings_i2, dense_embeddings_i2 = self.prompt_encoder(
                        points=prompt_temp2, boxes=None, masks=None
                    )
                    sparse_embeddings2.append(sparse_embeddings_i2)
                    dense_embeddings2.append(dense_embeddings_i2)
                
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

            low_res_masks2 = []
            iou_predictions2 = []
            for i in range(input_images.shape[0]):
                low_res_masks2_i, iou_predictions2_i, _ = self.mask_decoder2(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i],
                dense_prompt_embeddings=dense_embeddings[i],
                multimask_output=multimask_output  
            )
                low_res_masks2.append(low_res_masks2_i)
                iou_predictions2.append(iou_predictions2_i)
            
            low_res_masks2 = torch.cat(low_res_masks2, dim = 0)
            iou_predictions2 = torch.cat(iou_predictions2, dim = 0)

            low_res_masks2_2 = []
            iou_predictions2_2 = []
            for i in range(input_images.shape[0]):
                low_res_masks2_i2, iou_predictions2_i2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings2[i],
                dense_prompt_embeddings=dense_embeddings2[i],
                multimask_output=multimask_output  
            )
                low_res_masks2_2.append(low_res_masks2_i2)
                iou_predictions2_2.append(iou_predictions2_i2)
            
            low_res_masks2_2 = torch.cat(low_res_masks2_2, dim = 0)
            iou_predictions2_2 = torch.cat(iou_predictions2_2, dim = 0)
        
        elif prompt_idx == 1:  
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
        
            low_res_masks2, iou_predictions2, embed = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            points_prompt, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks2, image_size, embed_new, labels, scaling1)  
            points_prompt2, box_prompt2, mask_prompt2 = self.prompt_generate_random_fast(low_res_masks2, image_size, embed_new, labels, scaling2) 

            if prompt == 'point':
                sparse_embeddings = []
                dense_embeddings = []
                for i in range(input_images.shape[0]):
                    prompt_temp = (points_prompt[0][i].unsqueeze(0), points_prompt[1][i].unsqueeze(0))
                    sparse_embeddings_i, dense_embeddings_i = self.prompt_encoder(
                        points=prompt_temp, boxes=None, masks=None
                    )
                    sparse_embeddings.append(sparse_embeddings_i)
                    dense_embeddings.append(dense_embeddings_i)
                
                sparse_embeddings2 = []
                dense_embeddings2 = []
                for i in range(input_images.shape[0]):
                    prompt_temp2 = (points_prompt2[0][i].unsqueeze(0), points_prompt2[1][i].unsqueeze(0))
                    sparse_embeddings_i2, dense_embeddings_i2 = self.prompt_encoder(
                        points=prompt_temp2, boxes=None, masks=None
                    )
                    sparse_embeddings2.append(sparse_embeddings_i2)
                    dense_embeddings2.append(dense_embeddings_i2)
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

            low_res_masks1 = []
            iou_predictions1 = []
            for i in range(input_images.shape[0]):
                low_res_masks1_i, iou_predictions1_i, _ = self.mask_decoder1(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i],
                dense_prompt_embeddings=dense_embeddings[i],
                multimask_output=multimask_output  
            )
                low_res_masks1.append(low_res_masks1_i)
                iou_predictions1.append(iou_predictions1_i)
            
            low_res_masks1 = torch.cat(low_res_masks1, dim = 0)
            iou_predictions1 = torch.cat(iou_predictions1, dim = 0)

            low_res_masks1_2 = []
            iou_predictions1_2 = []
            for i in range(input_images.shape[0]):
                low_res_masks1_i2, iou_predictions1_i2, _ = self.mask_decoder1(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings2[i],
                dense_prompt_embeddings=dense_embeddings2[i],
                multimask_output=multimask_output  
            )
                low_res_masks1_2.append(low_res_masks1_i2)
                iou_predictions1_2.append(iou_predictions1_i2)
            
            low_res_masks1_2 = torch.cat(low_res_masks1_2, dim = 0)
            iou_predictions1_2 = torch.cat(iou_predictions1_2, dim = 0)
        
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
            )
            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )
            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )


        masks1 = self.postprocess_masks(
            low_res_masks1,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        masks2 = self.postprocess_masks(
            low_res_masks2,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        if prompt_idx != -1:
            if prompt_idx == 1:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'low_res_logits1_2': low_res_masks1_2,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2
                }
            else:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2,
                    'low_res_logits2_2': low_res_masks2_2
                }
        else:
            outputs = {
                'masks': masks1,
                'iou_predictions1': iou_predictions1,
                'low_res_logits1': low_res_masks1,
                'masks2': masks2,
                'iou_predictions2': iou_predictions2,
                'low_res_logits2': low_res_masks2
            }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions, _ = self.mask_decoder1(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def prompt_generate_random_fast(self, coarse_mask, img_size, image_embedding, labels, scaling = 8):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape

        coarse_mask_np = torch.argmax(coarse_mask.detach(), dim = 1)
        coarse_mask_np = F.interpolate(coarse_mask_np.unsqueeze(1).float(), (img_size, img_size), mode="nearest").squeeze(1)

        image_embedding = F.interpolate(image_embedding.detach(), (img_size, img_size), mode="bilinear").squeeze(1) 

        label_bs = labels.shape[0]
        coarse_mask_np = coarse_mask_np[label_bs:].long()
        coarse_mask_np = coarse_mask_np.detach().cpu()

        image_embedding = nn.functional.normalize(image_embedding, dim = 1)

        image_embedding_l = image_embedding[:label_bs]
        image_embedding_u = image_embedding[label_bs:]
        C = image_embedding_l.shape[1]

        ref_feat = []
        ref_label = []
        for i in range(0, label_bs):
            label_i = labels[i]
            image_embedding_l_i = image_embedding_l[i].permute(1, 2, 0)
            for j in range(0, num_class):
                if (label_i == j).sum() > 0:
                    feat_temp = image_embedding_l_i[label_i == j].mean(0)  
                    ref_feat.append(feat_temp)
                    ref_label.append(j)
        
        ref_feat = torch.stack(ref_feat, dim = 0) 
        ref_label = torch.tensor(ref_label) 

        out_temp = torch.einsum('bcpq, xc->bpqx', [image_embedding_u, ref_feat]) 
        sim, index = torch.topk(out_temp, k = 1, dim=-1, largest=True)
        # mask_guided = ref_label[index].squeeze(-1)  
        mask_guided = ref_label[index.to(ref_label.device)].squeeze(-1)
        
        mask_unlabel = -torch.ones_like(mask_guided)
        mask_unlabel[mask_guided == coarse_mask_np] = coarse_mask_np[mask_guided == coarse_mask_np]

        mask_guided = torch.cat([labels, mask_unlabel.to(labels.device)], dim = 0)  


        # points: BxNx2 tensor & boxes
        points_prompt = [[] for i in range(b)] 
        points_label = [[] for i in range(b)]
        points_prompt_random = None 
        for idx in range(b):  # iterate over each image
            for cls in range(num_class): # find points for each class
                # obtain the binary mask
                mask_cls = (mask_guided[idx] == cls)*1
                region = mask_cls.sum()
                if region > 0:
                    space = max(int(math.sqrt(region)/scaling),1)
                    cY_r, cX_r = torch.where(mask_cls==1)
                    # find the four corner coordinates
                    y0,x0 = cY_r.min(),cX_r.min()  
                    y1,x1 = cY_r.max(),cX_r.max()  
                    if (y0+space) >= (y1 - space + 1):  
                        y1 = y1 + space  
                        y0 = y0 - space
                    if (x0+space) >= (x1 - space + 1):  
                        x1 = x1 + space
                        x0 = x0 - space

                    y,x = torch.meshgrid(torch.arange(y0+space, y1 - space + 1, space), torch.arange(x0+space, x1 - space + 1, space))  
                    binary_mask = torch.zeros_like(mask_cls)
                    binary_mask[y,x] = 1
                    cY_r, cX_r = torch.where(binary_mask+mask_cls == 2)

                    if len(cY_r) == 0:
                        cY_r, cX_r = torch.where(mask_cls==1)
                        random_idx = np.random.randint(0, len(cX_r))
                        cX_r = cX_r[random_idx].unsqueeze(0)
                        cY_r = cY_r[random_idx].unsqueeze(0)

                    point_prompts = torch.stack([cX_r, cY_r], dim = -1)
                    points_prompt[idx].append(point_prompts)

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    
                    label_prompts = torch.ones_like(cY_r)*cls
                    points_label[idx].append(label_prompts)

            if len(points_prompt[idx]) == 0:
                print('mask_guided[idx].max(): ', mask_guided[idx].max())
                point_prompts = torch.zeros([1,2]).to(mask_guided[idx].device)
                label_prompts = torch.zeros([1]).to(mask_guided[idx].device)
                points_prompt[idx].append(point_prompts)
                points_label[idx].append(label_prompts)
            points_prompt[idx] = torch.cat(points_prompt[idx], dim = 0)
            points_label[idx] = torch.cat(points_label[idx], dim = 0)

        points_prompt = (points_prompt, points_label)  


        return points_prompt, None, None
