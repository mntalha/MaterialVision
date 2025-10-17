import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import open_clip
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mobileclip.modules.common.mobileone import reparameterize_model

class ImageTextDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer, train=True):
        self.df = df
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.train = train

    def __len__(self):
        return len(self.df)

    def list_to_image(self, img_list, size=224):
        """Convert a list to a 2D image of given size."""
        return np.array(json.loads(img_list)).reshape(size, size)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        image = self.list_to_image(row['image'])
        image = Image.fromarray(image).convert('RGB')
        image = self.preprocess(image)

        # Get text
        text = row['input']  # or row['caption'] depending on your data format
        
        # Prepare sample
        return {
            'image': image,
            'text': text,
            'caption': text  # keep original text for evaluation
        }

class MobileCLIPModel(nn.Module):
    def __init__(self, model_name='MobileCLIP2-S0', model_path=None):
        super().__init__()
        # Create base model and transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Reparameterize for inference
        self.model = reparameterize_model(self.model)
        
        # Freeze base model if needed
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images, text):
        # Process images
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Process text
        if isinstance(text, (list, tuple)):
            text = self.tokenizer(text)
        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    
    for batch in progress_bar:
        # Move data to device
        images = batch['image'].to(device)
        texts = batch['text']  # Keep as list/text for tokenizer
        
        # Forward pass
        image_features, text_features = model(images, texts)
        
        # Compute similarity
        logits = image_features @ text_features.T
        
        # Compute loss
        labels = torch.arange(len(images), device=device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(device)
            texts = batch['text']
            
            image_features, text_features = model(images, texts)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # Concatenate all features
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    # Compute similarity matrix
    similarity = (image_features @ text_features.T).cpu()
    
    # Compute metrics
    labels = torch.arange(len(similarity))
    loss_i = F.cross_entropy(similarity, labels)
    loss_t = F.cross_entropy(similarity.T, labels)
    loss = (loss_i + loss_t) / 2
    
    return loss.item()

def main():
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = './mobileclip2_s0.pt'  # Update with your model path
    
    # Load data
    train_df = pd.read_csv('../../data/alpaca_mbj_bandgap_train.csv')
    val_df = pd.read_csv('../../data/alpaca_mbj_bandgap_test.csv')
    
    # Create model
    model = MobileCLIPModel(model_path=MODEL_PATH)
    model = model.to(DEVICE)
    
    # Create datasets
    train_ds = ImageTextDataset(train_df, model.preprocess, model.tokenizer, train=True)
    val_ds = ImageTextDataset(val_df, model.preprocess, model.tokenizer, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch + 1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'checkpoints/best_mobileclip.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('checkpoints/loss.png')
    plt.close()

if __name__ == '__main__':
    # Create checkpoint directory
    Path('checkpoints').mkdir(exist_ok=True)
    main()