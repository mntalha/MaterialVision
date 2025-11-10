"""
Model loading functions for the MaterialVision webapp.

This module provides loading functions for different vision-language models:
- CLIPP-SciBERT (CLIPP_allenai)
- CLIPP-DistilBERT (CLIPP_bert) 
- MobileCLIP (Apple_MobileCLIP)
- BLIP (Salesforce)
"""

import sys
import os
from pathlib import Path
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, DistilBertTokenizer
import open_clip

# Get the current working directory and navigate to the correct paths
current_dir = Path.cwd()
if current_dir.name == 'webapp':
    webapp_dir = current_dir
    models_dir = webapp_dir.parent / 'models'
else:
    # If running from MaterialVision root or elsewhere
    webapp_dir = current_dir / 'webapp'
    models_dir = current_dir / 'models'

# Helper function to clear module cache
def clear_module_cache(module_names):
    """Clear specified modules from sys.modules cache"""
    for module_name in module_names:
        if module_name in sys.modules:
            del sys.modules[module_name]

# Import models with proper path management and cache clearing
def _import_clipp_scibert():
    clipp_path = str(models_dir / 'CLIPP_allenai')
    print(f"Adding to path: {clipp_path}")
    
    # Clear any cached modules that might conflict
    clear_module_cache(['training', 'config'])
    
    sys.path.insert(0, clipp_path)
    try:
        import training
        importlib.reload(training)  # Force reload
        from training import CLIPP, ImageTextDataset
        print("✅ Successfully imported CLIPP SciBERT")
        return CLIPP, ImageTextDataset
    except ImportError as e:
        print(f"❌ Error importing CLIPP SciBERT: {e}")
        return None, None
    finally:
        if clipp_path in sys.path:
            sys.path.remove(clipp_path)
        clear_module_cache(['training'])  # Clean up

def _import_clipp_distilbert():
    clipp_bert_path = str(models_dir / 'CLIPP_bert')
    print(f"Adding to path: {clipp_bert_path}")
    
    # Clear any cached modules that might conflict
    clear_module_cache(['training', 'config'])
    
    sys.path.insert(0, clipp_bert_path)
    try:
        import training
        import config
        importlib.reload(training)  # Force reload
        importlib.reload(config)    # Force reload
        from training import CLIPModel, ImageTextDataset
        from config import CFG
        print("✅ Successfully imported CLIPP DistilBERT")
        return CLIPModel, ImageTextDataset, CFG
    except ImportError as e:
        print(f"❌ Error importing CLIPP DistilBERT: {e}")
        return None, None, None
    finally:
        if clipp_bert_path in sys.path:
            sys.path.remove(clipp_bert_path)
        clear_module_cache(['training', 'config'])  # Clean up

def _import_apple_mobileclip():
    mobileclip_path = str(models_dir / 'Apple_MobileCLIP')
    print(f"Adding to path: {mobileclip_path}")
    
    # Clear any cached modules that might conflict
    clear_module_cache(['training', 'config'])
    
    sys.path.insert(0, mobileclip_path)
    try:
        import training
        importlib.reload(training)  # Force reload
        from training import AppleCLIPP, ImageTextDataset
        print("✅ Successfully imported MobileCLIP")
        return AppleCLIPP, ImageTextDataset
    except ImportError as e:
        print(f"❌ Error importing MobileCLIP: {e}")
        return None, None
    finally:
        if mobileclip_path in sys.path:
            sys.path.remove(mobileclip_path)
        clear_module_cache(['training'])  # Clean up

def _import_blip():
    blip_path = str(models_dir / 'Salesforce')
    print(f"Adding to path: {blip_path}")
    
    # Clear any cached modules that might conflict
    clear_module_cache(['train_blip'])
    
    sys.path.insert(0, blip_path)
    try:
        import train_blip
        importlib.reload(train_blip)  # Force reload
        from train_blip import BlipForRetrieval, ImageTextDataset
        print("✅ Successfully imported BLIP")
        return BlipForRetrieval, ImageTextDataset
    except ImportError as e:
        print(f"❌ Error importing BLIP: {e}")
        return None, None
    finally:
        if blip_path in sys.path:
            sys.path.remove(blip_path)
        clear_module_cache(['train_blip'])  # Clean up

# Get the model classes
CLIPPSciBERT, ImageTextDatasetSciBERT = _import_clipp_scibert()
CLIPPDistilBERT, ImageTextDatasetDistilBERT, CFG = _import_clipp_distilbert()
MobileCLIPModel, ImageTextDatasetMobileCLIP = _import_apple_mobileclip()
BlipForRetrieval, ImageTextDatasetBLIP = _import_blip()


#Create a dumb dataframe sample for dataset initialization
import pandas as pd
dummy_data = pd.DataFrame({'input': ['dummy_path1', 'dummy_path2'],
                            'text': ['dummy text 1', 'dummy text 2']})
# Initialize datasets to avoid errors during model loading
# --- Loading Functions ---

def load_clipp_scibert(checkpoint_path: str, device: str):
    """Load CLIPP model with SciBERT text encoder (CLIPP_allenai).
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, tokenizer, ImageTextDataset)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    dataset = ImageTextDatasetSciBERT(dummy_data, tokenizer=tokenizer)
    
    # Create model - defaults match the training script
    model = CLIPPSciBERT(
        vision_name='vit_base_patch16_224',
        text_name='allenai/scibert_scivocab_uncased',
        proj_dim=256
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval
    model.to(device)
    model.eval()
    
    # Add convenience methods for compatibility with app.py
    def get_image_features(images):
        with torch.no_grad():
            img_feats = model.vision(images)
            img_emb = model.img_proj(img_feats)
            return F.normalize(img_emb, dim=-1)
    
    def get_text_features(input_ids, attention_mask):
        with torch.no_grad():
            txt_out = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            txt_cls = txt_out[:, 0, :]
            txt_emb = model.txt_proj(txt_cls)
            return F.normalize(txt_emb, dim=-1)
    
    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    
    return model, tokenizer, dataset


def load_clipp_distilbert(checkpoint_path: str, device: str):
    """Load CLIPP model with DistilBERT text encoder.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, tokenizer, ImageTextDataset)
    """
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)
    
    # Load dataset
    dataset = ImageTextDatasetDistilBERT(dummy_data, tokenizer=tokenizer)
    # Create model
    model = CLIPPDistilBERT()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval
    model.to(device)
    model.eval()
    
    # Add convenience methods for compatibility with app.py
    def get_image_features(images):
        with torch.no_grad():
            image_features = model.image_encoder(images)
            image_embeddings = model.image_projection(image_features)
            return F.normalize(image_embeddings, dim=-1)
    
    def get_text_features(input_ids, attention_mask):
        with torch.no_grad():
            text_output = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = model.text_projection(text_output.last_hidden_state[:, 0, :])
            return F.normalize(text_embeddings, dim=-1)
    
    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    
    return model, tokenizer, dataset


def load_mobileclip(checkpoint_path: str, device: str):
    """Load MobileCLIP model (Apple_MobileCLIP).
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, tokenizer, ImageTextDataset)
    """
    # Load MobileCLIP-S2 tokenizer
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
    
    # Create model
    model = MobileCLIPModel(proj_dim=256)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval
    model.to(device)
    model.eval()
    
    # Load dataset    
    dataset = ImageTextDatasetMobileCLIP(dummy_data, tokenizer, model.preprocess, train=False)

    # Add convenience methods for compatibility with app.py
    def get_image_features(images):
        with torch.no_grad():
            img_emb = model.model.encode_image(images)
            img_emb = model.img_proj(img_emb)
            return F.normalize(img_emb, dim=-1)
    
    def get_text_features(input_ids, attention_mask=None):
        with torch.no_grad():
            # For MobileCLIP, input_ids should be the tokenized text
            txt_emb = model.model.encode_text(input_ids)
            txt_emb = model.txt_proj(txt_emb)
            return F.normalize(txt_emb, dim=-1)
    
    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    
    return model, tokenizer, dataset


def load_blip(checkpoint_path: str, device: str):
    """Load BLIP model for image-text retrieval (Salesforce).
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, processor, ImageTextDataset)
    """
    # Load processor
    model_name = "Salesforce/blip-itm-large-coco"
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Create model
    model = BlipForRetrieval.from_pretrained(model_name)
    
    # Move to device and set to eval
    model.to(device)
    model.eval()
    
    #Load dataset
    dataset = ImageTextDatasetBLIP(dummy_data, processor, train=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, processor, dataset