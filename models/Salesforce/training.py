import numpy as np
import json
import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

def list_to_image(img_list, size=224):
    """
    Convert a list to a 2D image of given size.
    """
    return np.array(json.loads(img_list)).reshape(size, size)

import re

def extract_formula_bandgap(text):
    formula_match = re.search(r'The chemical formula is ([A-Za-z0-9]+)', text)
    bandgap_match = re.search(r'mbj_bandgap value is ([0-9.]+)', text)
    formula = formula_match.group(1) if formula_match else None
    bandgap_str = bandgap_match.group(1) if bandgap_match else None
    if bandgap_str:
        bandgap_str = bandgap_str.strip().rstrip('.')
        bandgap = float(bandgap_str)
    else:
        bandgap = None
    return f"{formula} {bandgap}"

from PIL import Image

torch_preprocess = transforms.Compose([
    transforms.ToTensor(),

])

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

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
from torch import nn
from typing import Optional, Tuple
from torch.nn.functional import normalize

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



model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

ckpt = torch.load("/home/jipengsun/clipp/PlugIR/finetune/epoch34_.pth", weights_only=False, map_location="cpu")
state_dict = ckpt['model']
msg = model.load_state_dict(state_dict, strict=False)
model.to(device)

import pandas as pd
data_test = pd.read_csv('/home/jipengsun/clipp/dataset/alpaca_mbj_bandgap_test.csv') # Load your test data if needed

dataset = image_title_dataset(data_test, processor) # Use a subset for testing

# Further reduce batch size to minimize memory usage
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

# Clear GPU memory before starting the process
torch.cuda.empty_cache()

# Enable mixed precision to save memory
from torch.amp import autocast
import tqdm

corpus_vectors = []
corpus_ids = []

text_vectors = []

# Clear GPU memory before processing each batch
for batch in tqdm.tqdm(dataloader):
    torch.cuda.empty_cache()
    with autocast('cuda'):
        batch_vectors = F.normalize(model.get_image_features(batch[0].to(device)), dim=-1)
        corpus_vectors.append(batch_vectors)

        txts = processor(text=batch[1], padding=True, return_tensors='pt').to(device)
        txt_features = model.get_text_features(**txts)
        text_vectors.append(txt_features)

        corpus_ids.append(batch[2].int().to(device))

corpus_vectors = torch.cat(corpus_vectors)
text_vectors = torch.cat(text_vectors)
corpus_ids = torch.cat(corpus_ids)

# sort by id: important!
arg_ids = torch.argsort(corpus_ids)
corpus_vectors = corpus_vectors[arg_ids]
text_vectors = text_vectors[arg_ids]
corpus_ids = corpus_ids[arg_ids]

text = "he chemical formula is Li2GePd. The  mbj_bandgap value is 0.0."
text = processor(text=text, padding=True, return_tensors='pt').to(device)
text_features = model.get_text_features(**text)

# Ensure text_vectors and text_features are on the same device and shape
similarities = (text_features @ text_vectors.T).squeeze(0)  # shape: (num_texts,)
closest_idx = similarities.argmax().item()
closest_text = data_test.iloc[closest_idx]['input']
print("Closest text:", closest_text)

# scores: (num_texts, num_images)
scores = text_vectors @ corpus_vectors.T
top1 = np.mean([np.argmax(scores[i]) == i for i in range(scores.shape[0])])
top5 = np.mean([i in np.argsort(scores[i])[::-1][:5] for i in range(scores.shape[0])])
top10 = np.mean([i in np.argsort(scores[i])[::-1][:10] for i in range(scores.shape[0])])

print(f"Top-1: {top1:.4f}, Top-5: {top5:.4f}, Top-10: {top10:.4f}")
