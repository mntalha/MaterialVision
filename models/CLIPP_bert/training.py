"""
CLIPP training script with DistilBERT (lightweight BERT variant for materials science text-image matching).

This file provides:
- dataset parsing using existing `image` (JSON list) and `input` columns
- a dual-encoder model (ResNet for images, DistilBERT for text) with projection heads
- training and validation loops with checkpointing and loss plotting
"""

import json
import os
from pathlib import Path
import argparse
import logging
import math
import re

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from tqdm.auto import tqdm

from config import CFG

# Set up logging to both file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters and handlers
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Ensure checkpoints directory exists
log_dir = Path('./checkpoints')
log_dir.mkdir(exist_ok=True, parents=True)

# File handler
file_handler = logging.FileHandler(log_dir / 'training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def list_to_image(img_list, size=CFG.size):
    """Convert a list to a 2D image array."""
    if isinstance(img_list, str):
        arr = np.array(json.loads(img_list))
    else:
        arr = np.array(img_list)
    return arr.reshape(size, size)

# Image transforms
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG.size, CFG.size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG.size, CFG.size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def parse_chemical_formula(formula):
    """Parse a chemical formula into separated elements with counts.
    
    Args:
        formula: Chemical formula string like "Fe2O3"
        
    Returns:
        Formatted string like "2 Fe 3 O" or original formula if parsing fails
    """
    if not formula:
        return ""
    
    try:
        # Match element-count pairs: uppercase letter + optional lowercase + optional digits
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        if not matches:
            return formula
        
        result_parts = []
        for element, count in matches:
            # If no count specified, it's implicitly 1
            if not count:
                count = "1"
            result_parts.extend([count, element])
        
        return ' '.join(result_parts)
    except Exception:
        return formula


def extract_formula_bandgap(text):
    """Extract chemical formula and MBJ bandgap from a prompt text.

    Returns a compact caption string like: "2 Fe 3 O 1.23" or the original text if parsing fails.
    """
    if not isinstance(text, str):
        return str(text)
    formula_match = re.search(r'The chemical formula is ([A-Za-z0-9]+)', text)
    bandgap_match = re.search(r'mbj_bandgap value is ([0-9.]+)', text)
    formula = formula_match.group(1) if formula_match else None
    bandgap_str = bandgap_match.group(1) if bandgap_match else None
    if bandgap_str:
        try:
            bandgap = float(bandgap_str.strip().rstrip('.'))
        except Exception:
            bandgap = None
    else:
        bandgap = None
    if formula is None and bandgap is None:
        return text
    
    # Parse the chemical formula to separate elements
    parsed_formula = parse_chemical_formula(formula) if formula else ""
    return f"{parsed_formula} {bandgap if bandgap is not None else ''}".strip()

class ImageTextDataset(Dataset):
    """Dataset for image-text pairs."""
    def __init__(self, dataframe, tokenizer, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.train = train
        self.captions = [extract_formula_bandgap(t) for t in self.df['input'].tolist()]
        self.encoded = tokenizer(self.captions, padding='max_length', truncation=True, max_length=128)

    def prepare_caption(self, text):
        caption = extract_formula_bandgap(text)
        encoded = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids, attention_mask = torch.tensor(encoded['input_ids']), torch.tensor(encoded['attention_mask'])
        return caption, input_ids, attention_mask
    
    def prepare_image(self, path):
        img = Image.open(path).convert('RGB')
        img = VAL_TRANSFORMS(img)
        return img
     
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Process image
        img_json = self.df.loc[idx, 'image']
        img_arr = list_to_image(img_json)
        img = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
        img = TRAIN_TRANSFORMS(img) if self.train else VAL_TRANSFORMS(img)


        # text tokens
        input_ids = torch.tensor(self.encoded['input_ids'][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.encoded['attention_mask'][idx], dtype=torch.long)

        return {
            'image': img,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': self.captions[idx]
        }

class CLIPModel(nn.Module):
    """Dual-encoder model with ResNet image encoder and DistilBERT text encoder."""
    def __init__(self, temperature=CFG.temperature):
        super().__init__()
        self.temperature = temperature
        
        # Image encoder
        self.image_encoder = timm.create_model(
            CFG.model_name,
            pretrained=CFG.pretrained,
            num_classes=0
        )
        vision_dim = CFG.image_embedding

        # Text encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained(CFG.text_encoder_model)
        text_dim = CFG.text_embedding

        # Projection heads
        self.image_projection = nn.Sequential(
            nn.Linear(vision_dim, CFG.projection_dim),
            nn.LayerNorm(CFG.projection_dim),
            nn.Dropout(CFG.dropout)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, CFG.projection_dim),
            nn.LayerNorm(CFG.projection_dim),
            nn.Dropout(CFG.dropout)
        )

        # Freeze backbones if not trainable
        if not CFG.trainable:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # Get image features
        image_features = self.image_encoder(batch['image'])
        image_embeddings = self.image_projection(image_features)

        # Get text features (use CLS token)
        text_output = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        text_embeddings = self.text_projection(text_output.last_hidden_state[:, 0, :])

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity and loss
        logits = image_embeddings @ text_embeddings.t() / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

        return loss

def train_epoch(model, train_loader, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc='Train'):
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != 'caption'}
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Valid'):
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != 'caption'}
            loss = model(batch)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    # Load data
    train_df = pd.read_csv('../../data/alpaca_mbj_bandgap_train.csv')
    val_df = pd.read_csv('../../data/alpaca_mbj_bandgap_test.csv')
    
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    train_ds = ImageTextDataset(train_df, tokenizer, train=True)
    val_ds = ImageTextDataset(val_df, tokenizer, train=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )
    
    model = CLIPModel().to(CFG.device)
    
    # Optimizer with different learning rates for each component
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": list(model.image_projection.parameters()) + 
                  list(model.text_projection.parameters()),
         "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    
    # Training loop
    save_dir = Path('./checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(CFG.epochs):
        logger.info(f"Epoch: {epoch + 1}/{CFG.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = save_dir / 'best_clipp_bert.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved Best Model to {checkpoint_path}!")
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig(save_dir / 'loss.png')
        plt.close()

if __name__ == '__main__':
    main()