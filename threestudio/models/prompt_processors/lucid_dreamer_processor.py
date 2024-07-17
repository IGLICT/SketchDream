import json
import os
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *

import math

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

@threestudio.register("lucid-dreamer-prompt-processor")
class LucidDreamerPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # Set prompt
        text: str = "a DSLR of a realistic army green jacket"
        negative: str = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation.'
        inverse_text: str = ''
        pretrained_model_name_or_path: str = ''

    cfg: Config

    def configure(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        
        with torch.no_grad():
            def get_text_embeds(prompt,):
                inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
                embeddings = text_encoder(inputs.input_ids.to(text_encoder.device))[0]
                return embeddings
            embeddings = {}
            embeddings['default'] = get_text_embeds([self.cfg.text])
            embeddings['uncond'] = get_text_embeds([self.cfg.negative])

            for d in ['front', 'side', 'back']:
                embeddings[d] = get_text_embeds([f"{self.cfg.text}, {d} view"])
            embeddings['inverse_text'] = get_text_embeds(self.cfg.inverse_text)
        
        self.embeddings_2d = embeddings
        del text_encoder
        
    @torch.no_grad()
    def get_text_embeddings(
        self,
        elevations: Float[Tensor, "B"],
        azimuths: Float[Tensor, "B"],
    ): # -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        text_z_ = []
        weights_ = []
        text_z_inverse =torch.cat([self.embeddings_2d['uncond'], self.embeddings_2d['inverse_text']], dim=0) # [2,77,1024]
        B, = elevations.shape
        for i in range(B):
            #pred text_z
            azimuth = azimuths[i] % 360
            if azimuth > 180:
                azimuth = azimuth - 360
            text_z = [self.embeddings_2d['uncond']]
             
            if azimuth >= -90 and azimuth < 90:
                if azimuth >= 0:
                    r = 1 - azimuth / 90
                else:
                    r = 1 + azimuth / 90
                start_z = self.embeddings_2d['front']
                end_z = self.embeddings_2d['side']
            else:
                if azimuth >= 0:
                    r = 1 - (azimuth - 90) / 90
                else:
                    r = 1 + (azimuth + 90) / 90
                start_z = self.embeddings_2d['side']
                end_z = self.embeddings_2d['back']
            text_z.append(r * start_z + (1 - r) * end_z)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)
        
        text_embeddings = torch.stack(text_z_, dim=1)
        
        return text_embeddings, text_z_inverse
    
    @torch.no_grad()
    def get_text_embeddings_perpneg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
    ): # -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        text_z_ = []
        weights_ = []
        text_z_inverse =torch.cat([self.embeddings_2d['uncond'], self.embeddings_2d['inverse_text']], dim=0) # [2,77,1024]
        B, = elevation.shape
        for i in range(1, B):
            #pred text_z
            azimuth = azimuth[i] % 360
            if azimuth > 180:
                azimuth = azimuth - 360
            text_z = [self.embeddings_2d['uncond']]
             
            text_z_comp, weights = adjust_text_embeddings(self.embeddings_2d, azimuth, self.guidance_2d_opt)
            text_z.append(text_z_comp)
            weights_.append(weights)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)
        
        text_embeddings = torch.stack(text_z_, dim=1)
        text_embeddings_weights = torch.stack(weights_, dim=1)
        
        return text_embeddings, text_embeddings_weights, text_z_inverse
