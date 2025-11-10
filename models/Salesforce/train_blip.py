"""
BLIP training script for image-text retrieval (Salesforce/blip-itm-large-coco).

This file provides:
- dataset parsing using existing `image` (JSON list) and `input` columns
- BLIP dual-encoder model with projection heads for contrastive learning
- training and validation loops with checkpointing and optional wandb logging
- mixed precision training and gradient accumulation support

Notes:
- Ensure the CSV files contain an `image` column (JSON-serialized list/array) and an `input` text column.
- Install dependencies: torch, transformers, pandas, numpy, tqdm, wandb (optional)

Code Sample Usage:
python train_blip.py \
  --train_csv ../../data/alpaca_mbj_bandgap_train.csv \
  --val_csv ../../data/alpaca_mbj_bandgap_test.csv \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-5 \
  --save_dir checkpoints_blip

nohup python train_blip.py \
  --train_csv ../../data/alpaca_mbj_bandgap_train.csv \
  --val_csv ../../data/alpaca_mbj_bandgap_test.csv \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-5 \
  --save_dir checkpoints_blip \
  --amp \
  --accumulate_steps 2 &
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
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoProcessor, BlipForImageTextRetrieval
from tqdm import tqdm
from typing import Optional, Tuple

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


class BlipForRetrieval(BlipForImageTextRetrieval):
    """Extended BLIP model with helper methods for feature extraction."""
    
    def get_text_features(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        text_feat = F.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

        return text_feat

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat

class ImageTextDataset(Dataset):
    """Dataset that returns preprocessed image tensor and tokenized text.

    Expects a dataframe with columns: `image` (JSON list or list-like) and `input` (text).
    """
    def __init__(self, dataframe: pd.DataFrame, processor, train: bool = True, img_size: int = 224):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.train = train
        self.img_size = img_size
        # Precompute captions
        self.captions = [extract_formula_bandgap(t) for t in self.df['input'].tolist()]

    def prepare_caption(self, text):
        caption = extract_formula_bandgap(text)
        txts = self.processor(text=list(caption), padding=True, return_tensors='pt')
        input_ids = txts['input_ids']
        attention_mask = txts['attention_mask']
        return caption, input_ids, attention_mask

    def prepare_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.processor(images=img, return_tensors='pt')
        pixel_values = img['pixel_values'][0]
        return pixel_values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image
        img_json = self.df.loc[idx, 'image']
        img_arr = list_to_image(img_json, size=self.img_size)
        img = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
        
        # Use BLIP processor for image preprocessing
        processed = self.processor(images=img, return_tensors='pt')
        pixel_values = processed['pixel_values'][0]
        
        # text - we'll tokenize in the training loop to maintain flexibility
        caption = self.captions[idx]
        
        return {
            'pixel_values': pixel_values,
            'caption': caption,
            'id': self.df.loc[idx, 'id'] if 'id' in self.df.columns else idx
        }



def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    """Compute bidirectional contrastive loss between image and text embeddings."""
    logits = img_emb @ txt_emb.t() / temperature
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def train_epoch(model, dataloader, optimizer, device, temperature, processor, scaler=None, accumulate_steps=1, clip_grad_norm=None):
    """Run one training epoch with optional mixed precision and gradient accumulation.

    Args:
        model: BlipForRetrieval model
        dataloader: DataLoader
        optimizer: torch optimizer
        device: torch.device
        temperature: float for contrastive loss
        processor: AutoProcessor for text tokenization
        scaler: GradScaler for mixed precision (optional)
        accumulate_steps: int for gradient accumulation
        clip_grad_norm: float or None for gradient clipping
    """
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Train')):
        pixel_values = batch['pixel_values'].to(device)
        captions = batch['caption']
        
        # Tokenize text
        txts = processor(text=list(captions), padding=True, return_tensors='pt').to(device)
        input_ids = txts['input_ids']
        attention_mask = txts['attention_mask']

        # get normalized features
        if scaler is not None:
            with autocast():
                img_feats = model.get_image_features(pixel_values)
                txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                img_norm = F.normalize(img_feats, dim=-1)
                txt_norm = F.normalize(txt_feats, dim=-1)
                loss = contrastive_loss(img_norm, txt_norm, temperature)
        else:
            img_feats = model.get_image_features(pixel_values)
            txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            img_norm = F.normalize(img_feats, dim=-1)
            txt_norm = F.normalize(txt_feats, dim=-1)
            loss = contrastive_loss(img_norm, txt_norm, temperature)

        if not torch.isfinite(loss):
            logger.warning(f'Non-finite loss at batch {batch_idx} (loss={loss}). Skipping step.')
            continue

        # gradient accumulation: scale loss to keep effective lr stable
        loss = loss / accumulate_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        steps += 1
        if steps % accumulate_steps == 0:
            if scaler is not None:
                if clip_grad_norm and clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_norm and clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps
    
    # average over true number of optimizer steps
    num_steps = max(1, math.ceil(len(dataloader) / accumulate_steps))
    return total_loss / num_steps


def validate(model, dataloader, device, temperature, processor, amp=False):
    """Run validation epoch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Valid'):
            pixel_values = batch['pixel_values'].to(device)
            captions = batch['caption']
            
            # Tokenize text
            txts = processor(text=list(captions), padding=True, return_tensors='pt').to(device)
            input_ids = txts['input_ids']
            attention_mask = txts['attention_mask']

            if amp:
                with autocast():
                    img_feats = model.get_image_features(pixel_values)
                    txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                    img_norm = F.normalize(img_feats, dim=-1)
                    txt_norm = F.normalize(txt_feats, dim=-1)
                    loss = contrastive_loss(img_norm, txt_norm, temperature)
            else:
                img_feats = model.get_image_features(pixel_values)
                txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                img_norm = F.normalize(img_feats, dim=-1)
                txt_norm = F.normalize(txt_feats, dim=-1)
                loss = contrastive_loss(img_norm, txt_norm, temperature)
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
    parser.add_argument('--save_dir', type=str, default='checkpoints_blip')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 or None to disable)')
    parser.add_argument('--freeze_backbones', action='store_true', help='Freeze vision and text encoders and only train projection heads')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()

    # Prepare logging and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if _HAS_WANDB and not args.no_wandb:
        wandb.init(project='materialvision-blip', config=vars(args))

    # Load processor and pretrained model
    processor = AutoProcessor.from_pretrained('Salesforce/blip-itm-large-coco')
    # use the retrieval subclass that exposes feature helpers
    model = BlipForRetrieval.from_pretrained('Salesforce/blip-itm-large-coco')
    model.to(device)

    # load dataframes
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Create datasets using the improved dataset class
    train_ds = ImageTextDataset(train_df, processor, train=True)
    val_ds = ImageTextDataset(val_df, processor, train=False)

    # Use default collate for dict-based batches
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optionally freeze backbones and only train projection heads
    if args.freeze_backbones:
        logger.info('Freezing vision and text encoder parameters. Only projection heads will be trained.')
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    # Create optimizer only for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning('No trainable parameters found (all parameters frozen). Creating optimizer for all parameters instead.')
        trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    scaler = GradScaler() if args.amp else None

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
        logger.info(f'===== Epoch {epoch}/{args.epochs} =====')
        train_loss = train_epoch(model, train_loader, optimizer, device, args.temperature, processor,
                                 scaler=scaler, accumulate_steps=args.accumulate_steps, 
                                 clip_grad_norm=(args.clip_grad_norm if args.clip_grad_norm > 0 else None))
        val_loss = validate(model, val_loader, device, args.temperature, processor=processor, amp=args.amp)

        # record histories
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # plot and save loss curves into checkpoints directory
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            loss_plot_path = save_dir / 'loss.png'
            plt.savefig(str(loss_plot_path))
            plt.close()
            logger.info(f'Saved loss plot to {loss_plot_path}')
        except Exception as e:
            logger.warning(f'Could not save loss plot: {e}')

        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        
        if _HAS_WANDB and not args.no_wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / 'best_blip.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            logger.info(f'Saved new best checkpoint to {ckpt_path}')

    logger.info('Training finished')

if __name__ == '__main__':
    main()
