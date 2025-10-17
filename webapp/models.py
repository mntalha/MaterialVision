"""Model loading and inference utilities for MaterialVision."""

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
import timm
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoProcessor, 
    BlipForImageTextRetrieval
)

class CLIPPModel(torch.nn.Module):
    """Base CLIPP model class supporting both SciBERT and DistilBERT variants."""
    def __init__(self, vision_name='vit_base_patch16_224', text_name='allenai/scibert_scivocab_uncased', proj_dim=256):
        super().__init__()
        # Vision
        self.vision = timm.create_model(vision_name, pretrained=True, num_classes=0)
        vision_dim = self.vision.num_features

        # Text
        self.text_encoder = AutoModel.from_pretrained(text_name)
        text_dim = self.text_encoder.config.hidden_size

        # Projection heads
        self.img_proj = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, proj_dim), 
            torch.nn.LayerNorm(proj_dim)
        )
        self.txt_proj = torch.nn.Sequential(
            torch.nn.Linear(text_dim, proj_dim), 
            torch.nn.LayerNorm(proj_dim)
        )

    def get_image_features(self, images):
        img_feats = self.vision(images)
        img_emb = self.img_proj(img_feats)
        return normalize(img_emb, dim=-1)

    def get_text_features(self, input_ids, attention_mask):
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        txt_cls = txt_out[:, 0, :]
        txt_emb = self.txt_proj(txt_cls)
        return normalize(txt_emb, dim=-1)

class MobileCLIPModel(torch.nn.Module):
    """MobileCLIP model implementation."""
    def __init__(self):
        super().__init__()
        # Implement MobileCLIP architecture
        pass

    def get_image_features(self, images):
        # Implement image feature extraction
        pass

    def get_text_features(self, input_ids, attention_mask):
        # Implement text feature extraction
        pass

class BlipForRetrieval(BlipForImageTextRetrieval):
    """Extended BLIP model with direct feature access."""
    def get_text_features(self, input_ids, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_embeds = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=return_dict
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_image_features(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_feat = normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)
        return image_feat

def load_clipp_scibert(checkpoint_path, device='cuda'):
    """Load CLIPP-SciBERT model."""
    model = CLIPPModel(
        vision_name='vit_base_patch16_224',
        text_name='allenai/scibert_scivocab_uncased'
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Initialize vision backbone with pretrained weights
    model.vision.load_state_dict(
        timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).state_dict()
    )
    
    # Initialize text encoder with pretrained weights
    model.text_encoder.load_state_dict(
        AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').state_dict()
    )
    
    # Load only the projection heads from checkpoint
    proj_state_dict = {
        k: v for k, v in state_dict.items() 
        if k.startswith('img_proj') or k.startswith('txt_proj')
    }
    model.load_state_dict(proj_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    return model, tokenizer

def load_clipp_distilbert(checkpoint_path, device='cuda'):
    """Load CLIPP-DistilBERT model."""
    model = CLIPPModel(
        vision_name='vit_base_patch16_224',
        text_name='distilbert-base-uncased'
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Initialize vision backbone with pretrained weights
    model.vision.load_state_dict(
        timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).state_dict()
    )
    
    # Initialize text encoder with pretrained weights
    model.text_encoder.load_state_dict(
        AutoModel.from_pretrained('distilbert-base-uncased').state_dict()
    )
    
    # Load only the projection heads from checkpoint
    proj_state_dict = {
        k: v for k, v in state_dict.items() 
        if k.startswith('img_proj') or k.startswith('txt_proj')
    }
    model.load_state_dict(proj_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

def load_mobileclip(checkpoint_path, device='cuda'):
    """Load MobileCLIP model."""
    model = MobileCLIPModel()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # or appropriate tokenizer
    return model, tokenizer

def load_blip(checkpoint_path, device='cuda'):
    """Load BLIP model."""
    from transformers import AutoTokenizer, BlipProcessor
    
    model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    image_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-itm-large-coco")
    
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Try different possible state dict locations
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove any 'module.' prefix from state dict keys if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_path}: {str(e)}")
            print("Using default pretrained weights instead.")
    
    model = model.to(device)
    model.eval()
    
    # Create a wrapper class to handle both image and text processing
    class BLIPProcessor:
        def __init__(self, image_processor, tokenizer):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            
    processor = BLIPProcessor(image_processor, tokenizer)
    return model, processor