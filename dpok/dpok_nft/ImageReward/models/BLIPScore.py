'''
@File       :   BLIPScore.py
@Time       :   2023/02/19 20:48:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   BLIPScore.
* Based on BLIP code base
* https://github.com/salesforce/BLIP
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ImageReward.models.BLIP.blip import load_checkpoint
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class BLIPScore(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        
        self.preprocess = _transform(224)
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)


    def score(self, prompt, image_path):
        
        if (type(image_path).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image_path)
            return rewards
            
        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
        
        # image encode
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1)    
        
        # score
        rewards = torch.sum(torch.mul(txt_feature, image_features), dim=1, keepdim=True)
        
        return rewards.detach().cpu().numpy().item()


    def inference_rank(self, prompt, generations_list):
    
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
        
        txt_set = []
        img_set = []
        for generations in generations_list:
            # image encode
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1)    
            img_set.append(image_features)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()