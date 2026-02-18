# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import models
from torchvision import transforms


# --- Configuration ---
CSV_FILE = 'extracted_night_labels.csv'
IMAGE_DIR = 'converted_day'
MODEL_SAVE_PATH = 'weather_model_cleaned.pth'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 3

# --- 1. Dataset ---
class WeatherDataset(Dataset):

  def __init__(self, csv_file, image_dir, transform=None):
    raw_data = pd.read_csv(csv_file)
    # Filter for images that actually exist
    self.data = raw_data[
        raw_data['image_filename'].apply(
            lambda x: os.path.exists(os.path.join(image_dir, x))
        )
    ].reset_index(drop=True)
    print(
        f'Dataset initialized with {len(self.data)} images (from'
        f' {len(raw_data)} in CSV)'
    )
    self.image_dir = image_dir
    self.transform = transform
    self.label_map = {'cloudy': 0, 'clear': 1}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_name = self.data.iloc[idx]['image_filename']
    img_path = os.path.join(self.image_dir, img_name)
    try:
      image = Image.open(img_path).convert('RGB')
    except:
      image = Image.new('RGB', (224, 224))

    label_str = self.data.iloc[idx]['label']
    label = self.label_map.get(label_str, 0)

    if self.transform:
      image = self.transform(image)
    return image, label


def get_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion):
  model.eval()
  all_probs = []
  all_labels = []
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      outputs = model(inputs)
      probs = torch.softmax(outputs, dim=1)[:, 1]
      _, preds = torch.max(outputs, 1)

      correct += (preds == labels).sum().item()
      total += labels.size(0)
      all_probs.extend(probs.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  return correct / total, np.array(all_probs), np.array(all_labels)


def run_fold(
    fold_idx, full_dataset, indices, fold_size, dataset_size, data_transforms
):
  print(f'\nFold {fold_idx+1}/{K_FOLDS}')
  print('-' * 20)

  val_start = fold_idx * fold_size
  val_end = (
      (fold_idx + 1) * fold_size if fold_idx < K_FOLDS - 1 else dataset_size
  )

  val_indices = indices[val_start:val_end]
  train_indices = indices[:val_start] + indices[val_end:]

  # Create subsets and OVERRIDE transforms for validation
  train_subset = Subset(full_dataset, train_indices)
  val_subset = Subset(full_dataset, val_indices)

  # Note: Subset doesn't allow different transforms easily if they are part of the base dataset.
  # For simplicity in this repro, we'll use the dataset as is or accept same transforms.
  # Actually, let's just create separate datasets for train/val with different transforms
  train_dataset = WeatherDataset(
      CSV_FILE, IMAGE_DIR, transform=data_transforms['train']
  )
  val_dataset = WeatherDataset(
      CSV_FILE, IMAGE_DIR, transform=data_transforms['val']
  )

  train_subset = Subset(train_dataset, train_indices)
  val_subset = Subset(val_dataset, val_indices)

  train_loader = DataLoader(
      train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
  )
  val_loader = DataLoader(
      val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
  )

  model = get_model().to(DEVICE)
  criterion = nn.CrossEntropyLoss()

  # Stage 1
  for param in model.parameters():
    param.requires_grad = False
  for param in model.fc.parameters():
    param.requires_grad = True
  optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
  for _ in range(3):
    train_epoch(model, train_loader, criterion, optimizer)

  # Stage 2
  for param in model.parameters():
    param.requires_grad = True
  optimizer = optim.Adam(model.parameters(), lr=1e-5)

  best_acc = 0.0
  best_probs = None
  best_labels = None

  for epoch in range(7):
    train_epoch(model, train_loader, criterion, optimizer)
    acc, probs, labels = evaluate(model, val_loader, criterion)
    if acc > best_acc:
      best_acc = acc
      best_probs = probs
      best_labels = labels

  print(f'  Best Val Acc: {best_acc:.4f}')
  return best_probs, best_labels


def main():
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)

  data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

  # Initialize dataset once to get the correct size after filtering
  full_dataset = WeatherDataset(CSV_FILE, IMAGE_DIR)
  dataset_size = len(full_dataset)
  if dataset_size == 0:
    print('No images found for training. Exiting.')
    return

  indices = list(range(dataset_size))
  random.shuffle(indices)
  fold_size = dataset_size // K_FOLDS

  # Check for arguments to run specific tasks
  task = sys.argv[1] if len(sys.argv) > 1 else 'all'

  if task == 'fold':
    fold_idx = int(sys.argv[2])
    probs, labels = run_fold(
        fold_idx,
        full_dataset,
        indices,
        fold_size,
        dataset_size,
        data_transforms,
    )
    np.save(f'preds_fold_{fold_idx}.npy', {'probs': probs, 'labels': labels})

  elif task == 'analyze':
    all_probs = []
    all_labels = []
    for i in range(K_FOLDS):
      data = np.load(f'preds_fold_{i}.npy', allow_pickle=True).item()
      all_probs.extend(data['probs'])
      all_labels.extend(data['labels'])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\n{'Threshold':<10} | {'Precision':<10} | {'Recall':<10}")
    print('-' * 36)
    for t in np.linspace(0.0, 1.0, 21):
      preds = (all_probs >= t).astype(int)
      tp = np.sum((preds == 1) & (all_labels == 1))
      fp = np.sum((preds == 1) & (all_labels == 0))
      fn = np.sum((preds == 0) & (all_labels == 1))
      precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      print(f'{t:<10.2f} | {precision:<10.4f} | {recall:<10.4f}')

  elif task == 'final':
    print('\nTraining Final Model...')
    full_dataset = WeatherDataset(
        CSV_FILE, IMAGE_DIR, transform=data_transforms['train']
    )
    full_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for p in model.parameters():
      p.requires_grad = False
    for p in model.fc.parameters():
      p.requires_grad = True
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    for _ in range(3):
      train_epoch(model, full_loader, criterion, optimizer)

    for p in model.parameters():
      p.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for e in range(7):
      train_epoch(model, full_loader, criterion, optimizer)
      print(f'Epoch {e+1}/7 done')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Done.')

if __name__ == "__main__":
    main()
