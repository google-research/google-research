import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class RarityScore(nn.Module):
    def __init__(self, vit_model_weights_path):
        super().__init__()
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit_model.load_state_dict(torch.load(vit_model_weights_path)) # weights of our trained ViT model
        self.vit_model.to(self.device) # we have logits 
        #self.softmax = nn.Softmax(dim=1) # logits into probs
        self.sigmoid = torch.nn.functional.sigmoid()

        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.rewards = []

    def preprocess_image(self,image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image
    
    def compute_reward(self, image_path):
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.vit_model(image)
            logit = output.logit
        probabilities = self.sigmoid(logit)
        reward = probabilities[:,1].item()
        
        self.rewards.append(reward)

        return reward
    
    def normalize_reward(self):
        
        mean_reward = torch.tensor(self.rewards).mean().item()
        std_reward = torch.tensor(self.rewards).std().item()

        # normalized_rewards = []
        # for reward in self.rewards:
        #     normalized_reward = (reward - mean_reward) / std_reward
        #     normalized_rewards.append(normalized_reward)
        # self.rewards = normalized_reward
        self.rewards = [(reward - mean_reward) / std_reward for reward in self.rewards]



