"""
CLIPP training script (CLIP-style contrastive learning tuned for materials STEM data).

This file provides:
- dataset parsing using existing `image` (JSON list) and `input` columns
- a simple dual-encoder model (ViT for images, SciBERT for text) with projection heads
- training and validation loops with checkpointing and optional wandb logging

Notes:
- Ensure the CSV files contain an `image` column (JSON-serialized list/array) and an `input` text column.
- Install dependencies: torch, torchvision, timm, transformers, pandas, numpy, tqdm, wandb (optional)

Code Sample Usage:
nohup python training.py \
  --train_csv ../../data/alpaca_mbj_bandgap_train.csv \
  --val_csv ../../data/alpaca_mbj_bandgap_test.csv \
  --epochs 10 \
  --batch_size 16 \
  --lr 2e-5 \
  --save_dir checkpoints &
"""

import json
import re
import os
from pathlib import Path
import argparse
import logging
import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def list_to_image(img_list, size=224):
    """Convert a JSON list (string or list) to a 2D image array.

    Args:
        img_list: JSON string or python list containing pixel values
        size: output image side (image is reshaped to size x size)
    """
    if isinstance(img_list, str):
        arr = np.array(json.loads(img_list))
    else:
        arr = np.array(img_list)
    # If flattened length doesn't match, try flexible reshape
    if arr.size == size * size:
        return arr.reshape(size, size)
    # if already shaped
    if arr.ndim == 2 and arr.shape[0] == size and arr.shape[1] == size:
        return arr
    # fallback: attempt to reshape by inferring size
    side = int(math.sqrt(arr.size))
    return arr.reshape(side, side)


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


# Image transforms
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageTextDataset(Dataset):
    """Dataset that returns preprocessed image tensor and raw caption string and tokenized ids.

    Expects a dataframe with columns: `image` (JSON list or list-like) and `input` (text).
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer, train: bool = True, img_size: int = 224):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.train = train
        self.img_size = img_size
        # Precompute captions
        self.captions = [extract_formula_bandgap(t) for t in self.df['input'].tolist()]
        # Tokenize (we will convert to tensors in __getitem__ to keep Dataset picklable)
        self.encoded = tokenizer(self.captions, padding='max_length', truncation=True, max_length=128)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image
        img_json = self.df.loc[idx, 'image']
        img_arr = list_to_image(img_json, size=self.img_size)
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


class CLIPP(nn.Module):
    """Dual-encoder model: ViT image encoder + transformer text encoder + projection heads."""
    def __init__(self, vision_name='vit_base_patch16_224', text_name='allenai/scibert_scivocab_uncased', proj_dim=256):
        super().__init__()
        # Vision
        self.vision = timm.create_model(vision_name, pretrained=True, num_classes=0, in_chans=3)
        vision_dim = self.vision.num_features

        # Text
        self.text_encoder = AutoModel.from_pretrained(text_name)
        text_dim = self.text_encoder.config.hidden_size

        # Projection heads
        self.img_proj = nn.Sequential(nn.Linear(vision_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.txt_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.vision(images)  # (B, vision_dim)
        img_emb = self.img_proj(img_feats)

        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        txt_cls = txt_out[:, 0, :]
        txt_emb = self.txt_proj(txt_cls)

        img_norm = F.normalize(img_emb, dim=-1)
        txt_norm = F.normalize(txt_emb, dim=-1)
        return img_norm, txt_norm


def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    logits = img_emb @ txt_emb.t() / temperature
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def train_epoch(model, dataloader, optimizer, device, temperature, clip_grad_norm=None):
    """Run one training epoch with optional gradient clipping and basic diagnostics.

    Args:
        model: nn.Module
        dataloader: DataLoader
        optimizer: torch optimizer
        device: torch.device
        temperature: float
        clip_grad_norm: float or None - if set, clip gradients to this norm before optimizer.step()
    """
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Train')):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # debug prints removed

        img_emb, txt_emb = model(images, input_ids, attention_mask)
        loss = contrastive_loss(img_emb, txt_emb, temperature)

        # Basic numerical stability check
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at batch {batch_idx} (loss={loss}). Skipping step.")
            continue

        optimizer.zero_grad()
        loss.backward()

        # optional gradient clipping
        if clip_grad_norm is not None and clip_grad_norm > 0:
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            logger.debug(f"Grad norm: {total_norm:.4f}")

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device, temperature):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Valid'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_emb, txt_emb, temperature)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='../../data/alpaca_mbj_bandgap_train.csv')
    parser.add_argument('--val_csv', type=str, default='../../data/alpaca_mbj_bandgap_test.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 or None to disable)')
    parser.add_argument('--freeze_backbones', action='store_true', help='Freeze vision and text encoders and only train projection heads')
    # debug flag removed
    args = parser.parse_args()

    # Prepare logging and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if _HAS_WANDB and not args.no_wandb:
        wandb.init(project='materialvision-clipp', config=vars(args))

    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # (inspection helpers removed)

    train_ds = ImageTextDataset(train_df, tokenizer, train=True)
    val_ds = ImageTextDataset(val_df, tokenizer, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = CLIPP(proj_dim=args.proj_dim).to(device)

    # Optionally freeze pretrained backbones and only train projection heads
    if args.freeze_backbones:
        logger.info('Freezing vision and text encoder parameters. Only projection heads will be trained.')
        for param in model.vision.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    # Create optimizer only for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning('No trainable parameters found (all parameters frozen). Creating optimizer for all parameters instead.')
        trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # (inspection helpers removed) - continue to training

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Configure file logging inside the save directory so logs are colocated with checkpoints
    log_file = save_dir / 'train.log'
    try:
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        # Avoid adding duplicate file handlers
        existing_filenames = [getattr(h, 'baseFilename', None) for h in logger.handlers]
        if str(log_file) not in existing_filenames:
            logger.addHandler(fh)
        logger.info(f'Logging to {log_file}')
    except Exception as e:
        logger.warning(f'Could not create file handler for logging at {log_file}: {e}')

    best_val = float('inf')
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, args.epochs + 1):
        logger.info(f'Epoch {epoch}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, optimizer, device, args.temperature, clip_grad_norm=(args.clip_grad_norm if args.clip_grad_norm > 0 else None))
        val_loss = validate(model, val_loader, device, args.temperature)

        # record histories
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # plot and save loss curves into checkpoints directory
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='train')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            loss_plot_path = save_dir / 'loss.png'
            plt.tight_layout()
            plt.savefig(str(loss_plot_path))
            plt.close()
            logger.info(f'Saved loss plot to {loss_plot_path}')
        except Exception as e:
            logger.warning(f'Could not save loss plot: {e}')

        logger.info(f'Epoch {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
        if _HAS_WANDB and not args.no_wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / 'best_clipp.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}, ckpt_path)
            logger.info(f'Saved best model to {ckpt_path}')

    logger.info('Training finished')


if __name__ == '__main__':
    main()