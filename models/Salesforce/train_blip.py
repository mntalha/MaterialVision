"""Training script for BLIP image-text retrieval (Salesforce/blip-itm-large-coco).

Usage example:
python train_blip.py --train_csv ../../data/alpaca_mbj_bandgap_train.csv \
    --val_csv ../../data/alpaca_mbj_bandgap_test.csv --epochs 10 --batch_size 32 --lr 1e-5 --save_dir checkpoints_blip

python train_blip.py --train_csv ../../data/alpaca_mbj_bandgap_train.csv \
    --val_csv ../../data/alpaca_mbj_bandgap_test.csv --epochs 10 --batch_size 16 --lr 1e-5 --save_dir checkpoints_blip --amp --accumulate_steps 2

This script trains a retrieval-style contrastive objective between BLIP image and text encoders.
"""
import argparse
from pathlib import Path
import logging
import math
import json
from torch.nn.functional import normalize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
from torch import nn
from typing import Optional, Tuple
from torch.nn.functional import normalize

from torch.cuda.amp import autocast, GradScaler

from transformers import AutoProcessor, BlipForImageTextRetrieval

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BlipForRetrieval(BlipForImageTextRetrieval):
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

        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

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

        image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def list_to_image(img_list, size=224):
    """
    Convert a list to a 2D image of given size.
    """
    return np.array(json.loads(img_list)).reshape(size, size)


# Define a custom dataset
class image_title_dataset():
    def __init__(self, dataframe, processor):
        # Tokenize text
        self.text  = dataframe["input"]
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        # Preprocess image using CLIP's preprocessing function
        image = list_to_image(self.dataframe["image"][idx])
        image = Image.fromarray(image).convert("RGB") 
        image = self.processor(image)['pixel_values'][0] # Preprocess the image
        text_ = self.text[idx]
        id = self.dataframe["id"][idx]
        return image, text_, id



def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    logits = img_emb @ txt_emb.t() / temperature
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def train_epoch(model, dataloader, optimizer, device, temperature, processor, scaler=None, accumulate_steps=1, clip_grad_norm=None):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()
    for batch in tqdm(dataloader, desc='Train'):
        # Support two batch formats:
        # 1) dict with keys 'pixel_values', 'input_ids', 'attention_mask'
        # 2) tuple/list like (image_tensor, list_of_texts, ids_tensor)
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        else:
            # default collate yields: (images_tensor, list_of_texts, ids_tensor)
            pixel_values = batch[0].to(device)
            texts = batch[1]
            txts = processor(text=list(texts), padding=True, return_tensors='pt').to(device)
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
            logger.warning('Non-finite loss encountered, skipping step')
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
    return total_loss / (num_steps)


def validate(model, dataloader, device, temperature, processor=None, amp=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Valid'):
            if isinstance(batch, dict):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            else:
                pixel_values = batch[0].to(device)
                texts = batch[1]
                if processor is None:
                    raise RuntimeError('processor is required to tokenize validation texts when using tuple batches')
                txts = processor(text=list(texts), padding=True, return_tensors='pt').to(device)
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
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--save_dir', type=str, default='checkpoints_blip')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')



    # Load processor and pretrained model
    processor = AutoProcessor.from_pretrained('Salesforce/blip-itm-large-coco')
    # use the retrieval subclass that exposes feature helpers
    model = BlipForRetrieval.from_pretrained('Salesforce/blip-itm-large-coco')
    model.to(device)

    # load dataframes
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # The repository contains `image_title_dataset` which returns (image_tensor, text, id)
    # that is already preprocessed via the processor. Use that to match your existing dataset code.
    train_ds = image_title_dataset(train_df, processor)
    val_ds = image_title_dataset(val_df, processor)

    # Use default collate so batches are tuples: (images_tensor, list_texts, ids_tensor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optionally freeze backbones
    if args.freeze_backbones:
        for p in model.parameters():
            p.requires_grad = False
        # keep projection layers trainable if they exist
        for n, p in model.named_parameters():
            if 'vision_proj' in n or 'text_proj' in n or 'cls' in n or 'proj' in n:
                p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    scaler = GradScaler() if args.amp else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

        # --- Logging to file ---
    log_file = save_dir / 'train.log'
    try:
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        if str(log_file) not in [getattr(h, 'baseFilename', None) for h in logger.handlers]:
            logger.addHandler(fh)
        logger.info(f'Logging to {log_file}')
    except Exception as e:
        logger.warning(f'Could not create file handler for logging at {log_file}: {e}')

    best_val = float('inf')
    train_loss_history, val_loss_history = [], []

    for epoch in range(1, args.epochs + 1):
        logger.info(f'===== Epoch {epoch}/{args.epochs} =====')
        train_loss = train_epoch(model, train_loader, optimizer, device, args.temperature, processor,
                                 scaler=scaler, accumulate_steps=args.accumulate_steps, clip_grad_norm=args.clip_grad_norm)
        val_loss = validate(model, val_loader, device, args.temperature, processor=processor, amp=args.amp)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        # Save loss plot
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Train Loss')
            plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Val Loss')
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


if __name__ == '__main__':
    main()
