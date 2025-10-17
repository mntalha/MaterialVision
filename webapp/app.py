import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import json
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from models import (
    load_clipp_scibert,
    load_clipp_distilbert,
    load_mobileclip,
    load_blip
)

# streamlit run app.py

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
MODEL_PATHS = {
    'CLIPP-SciBERT': '../models/CLIPP_allenai/checkpoints/best_clipp.pth',
    'CLIPP-DistilBERT': '../models/CLIPP_bert/checkpoints/best_clipp_bert.pth',
    'MobileCLIP': '../models/Apple_MobileCLIP/checkpoints/best_mobileclip.pth',
    'BLIP': None  # Using default pretrained weights for BLIP
}


def list_to_image(img_list, size=224):
    """Convert a JSON list to a 2D image."""
    return np.array(json.loads(img_list)).reshape(size, size)

def extract_formula_bandgap(text):
    import re
    formula_match = re.search(r'The chemical formula is ([A-Za-z0-9]+)', text)
    bandgap_match = re.search(r'mbj_bandgap value is ([0-9.]+)', text)
    formula = formula_match.group(1) if formula_match else None
    bandgap_str = bandgap_match.group(1).rstrip('.') if bandgap_match else None  # strip trailing dot
    bandgap = float(bandgap_str) if bandgap_str else None 
    return formula, bandgap



# --- Dataset class ---
class ImageTextDataset():
    def __init__(self, dataframe, processor, model_name):
        self.dataframe = dataframe
        self.texts = dataframe["input"]
        self.processor = processor
        self.model_name = model_name
        
        # Define image transforms for CLIPP models
        self.clipp_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Load and convert image
        image = list_to_image(self.dataframe["image"][idx])
        image = Image.fromarray(image).convert("RGB")
        
        # Process image based on model type
        if self.model_name == 'BLIP':
            # For BLIP, use image_processor specifically for images
            processed = self.processor.image_processor(images=image, return_tensors="pt")
            image_tensor = processed['pixel_values'][0]  # Remove batch dimension
        else:  # CLIPP models and MobileCLIP
            image_tensor = self.clipp_transforms(image)
        
        text_ = self.texts[idx]
        id_ = self.dataframe["id"][idx]
        formula, bandgap = extract_formula_bandgap(text_)
        return image_tensor, text_, id_, formula, bandgap
    



# Load model
@st.cache_resource
def load_selected_model(model_name):
    """Load the selected model and its processor/tokenizer."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")
    
    checkpoint_path = MODEL_PATHS[model_name]
    
    if model_name == 'CLIPP-SciBERT':
        model, processor = load_clipp_scibert(checkpoint_path, device)
    elif model_name == 'CLIPP-DistilBERT':
        model, processor = load_clipp_distilbert(checkpoint_path, device)
    elif model_name == 'MobileCLIP':
        model, processor = load_mobileclip(checkpoint_path, device)
    else:  # BLIP
        model, processor = load_blip(checkpoint_path, device)
    
    return model, processor




import tqdm

# --- Load dataset ---
@st.cache_resource
def load_dataset(path='../data/alpaca_mbj_bandgap_test.csv', _processor=None, _model_name=None):
    data_test = pd.read_csv(path)[:100]
    dataset = ImageTextDataset(data_test, _processor, _model_name)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,  # Reduced batch size
        shuffle=False,
        num_workers=2,  # Use multiple workers for data loading
        pin_memory=True  # Speed up data transfer to GPU
    )
    return data_test, dataset, dataloader

# Initialize with default model
if 'model_name' not in st.session_state:
    st.session_state.model_name = "CLIPP-SciBERT"
    st.session_state.model, st.session_state.processor = load_selected_model(st.session_state.model_name)

# Load dataset with the current model's processor
data_test, dataset, dataloader = load_dataset(
    _processor=st.session_state.processor,
    _model_name=st.session_state.model_name
)



@st.cache_resource
def load_features():
    corpus_vectors, text_vectors, corpus_ids = [], [], []
    formulas, bandgaps = [], []
    
    # Set smaller batch size for feature extraction
    batch_size = 16
    n_samples = len(dataset)
    
    for i in tqdm.tqdm(range(0, n_samples, batch_size)):
        batch_indices = range(i, min(i + batch_size, n_samples))
        batch = [dataset[j] for j in batch_indices]
        
        # Collate batch manually
        imgs = torch.stack([item[0] for item in batch]).to(device)
        texts = [item[1] for item in batch]
        ids = [item[2] for item in batch]
        current_formulas = [item[3] for item in batch]
        current_bandgaps = [item[4] for item in batch]
        
        # Process in smaller chunks with memory cleanup
        with torch.no_grad():  # Disable gradient computation
            if st.session_state.model_name == 'BLIP':
                # Process text input
                text_inputs = st.session_state.processor.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Move tokenizer outputs to device
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                
                img_features = F.normalize(st.session_state.model.get_image_features(imgs), dim=-1)
                txt_features = F.normalize(st.session_state.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ), dim=-1)
            else:  # CLIPP models
                txts = st.session_state.processor(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Move required tensors to device and remove token_type_ids
                input_ids = txts['input_ids'].to(device)
                attention_mask = txts['attention_mask'].to(device)
                img_features = F.normalize(st.session_state.model.get_image_features(imgs), dim=-1)
                txt_features = F.normalize(
                    st.session_state.model.get_text_features(input_ids, attention_mask),
                    dim=-1
                )
            
            # Move results to CPU but keep as torch tensors
            img_features = img_features.cpu()
            txt_features = txt_features.cpu()
            
            corpus_vectors.append(img_features)
            text_vectors.append(txt_features)
            corpus_ids.extend(ids)
            formulas.extend(current_formulas)
            bandgaps.extend(current_bandgaps)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Combine results and convert to correct types
    corpus_vectors = torch.cat(corpus_vectors)
    text_vectors = torch.cat(text_vectors)
    corpus_ids = torch.tensor(corpus_ids)
    formulas = np.array(formulas)
    bandgaps = torch.tensor(bandgaps, dtype=torch.float32)
    
    # Sort all arrays
    sort_indices = torch.argsort(corpus_ids)
    corpus_vectors = corpus_vectors[sort_indices]
    text_vectors = text_vectors[sort_indices]
    corpus_ids = corpus_ids[sort_indices]
    formulas = formulas[sort_indices.cpu().numpy()]
    bandgaps = bandgaps[sort_indices]
    
    # Move tensors to device
    corpus_vectors = corpus_vectors.to(device)
    text_vectors = text_vectors.to(device)
    corpus_ids = corpus_ids.to(device)
    bandgaps = bandgaps.to(device)
    
    return corpus_vectors, text_vectors, corpus_ids, formulas, bandgaps

corpus_vectors, text_vectors, corpus_ids, formulas, bandgaps = load_features()



# Streamlit UI
st.title("🔍 Materials Science Retrieval App")

# Model selection
model_name = st.selectbox(
    "Select Model",
    ["CLIPP-SciBERT", "CLIPP-DistilBERT", "MobileCLIP", "BLIP"],
    help="Choose the model to use for retrieval"
)

# Load selected model
try:
    model, processor = load_selected_model(model_name)
    st.success(f"Successfully loaded {model_name}")
except Exception as e:
    st.error(f"Error loading {model_name}: {str(e)}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Text-to-Image", "Image-to-Text", "Metrics", "Filter by Bandgap"])

with tab1:
    print("Tab 1: Text-to-Image Retrieval")
    st.header("Text-to-Image & Text-to-Text Retrieval")
    text_query = st.text_input(
        "Enter a materials-related description:", 
        value="The chemical formula is UGe2Pt2. The mbj_bandgap value is 0.0."
    )
    k = st.slider("Number of top results to show:", 1, 10, 3)

    if st.button("Search Text"):
        st.write("Searching...")
        print("Searching for text...")
        with torch.no_grad():
            if st.session_state.model_name == 'BLIP':
                # Handle BLIP text processing using our custom processor class
                text_inputs = st.session_state.processor.tokenizer(
                    text_query,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Move tokenizer outputs to device
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                text_feat = F.normalize(model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ), dim=-1)
            else:  # CLIPP models
                text_input = processor(
                    text_query,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Move required tensors to device and remove token_type_ids
                input_ids = text_input['input_ids'].to(device)
                attention_mask = text_input['attention_mask'].to(device)
                text_feat = F.normalize(
                    model.get_text_features(input_ids, attention_mask),
                    dim=-1
                )

            # Search over the full dataset (no bandgap filter)
            sims_img = F.cosine_similarity(text_feat, corpus_vectors)
            sims_txt = F.cosine_similarity(text_feat, text_vectors)
            topk_img = torch.topk(sims_img, k)
            topk_txt = torch.topk(sims_txt, k)

            # Display results in columns
            cols = st.columns(3)

            with cols[0]:
                st.subheader("Top Matching Texts")
                for i in topk_txt.indices:
                    idx = i.item()
                    st.markdown(f"**{data_test.iloc[idx]['input']}**  \nScore: {sims_txt[i]:.3f}")

            with cols[1]:
                st.subheader("Top Matching Images")
                for i in topk_img.indices:
                    idx = i.item()
                    img = list_to_image(data_test.iloc[idx]["image"])
                    img = (img - img.min()) / (img.max() - img.min())
                    st.image(
                        img, 
                        caption=f"Formula: {formulas[idx]}, Score: {sims_img[i]:.3f}"
                    )

            with cols[2]:
                st.subheader("Download Results")
                top_results = data_test.iloc[topk_img.indices.cpu().numpy()]
                csv = top_results.to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="top_results.csv")


with tab2:
    print("Tab 2: Image-to-Text Retrieval")
    st.header("Image-to-Text Retrieval")
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    k_img2txt = st.slider("Top results to show:", 1, 10, 3, key="img2txt_slider")

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        if st.session_state.model_name == 'BLIP':
            # For BLIP, use image_processor specifically for images
            processed = st.session_state.processor.image_processor(images=img, return_tensors="pt")
            img_tensor = processed['pixel_values'].to(device)
        else:
            # For CLIPP models, use the transform directly
            img_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img).unsqueeze(0).to(device)
            
        with torch.no_grad():
            img_feat = F.normalize(model.get_image_features(img_tensor), dim=-1)
            sims = F.cosine_similarity(img_feat, text_vectors)
            topk = torch.topk(sims, k_img2txt)
        st.subheader("Top matching texts for uploaded image")
        for i in topk.indices:
            st.markdown(f"{data_test.iloc[i.item()]['input']}  \nScore: {sims[i]:.3f}")
            img_arr = list_to_image(data_test.iloc[i.item()]["image"])
            img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
            st.image(img_arr, caption=f"Formula: {formulas[i.item()]}, Bandgap: {bandgaps[i.item()]:.3f}")

with tab3:
    print("Tab 3: Metrics")
    st.header("Retrieval Metrics")
    # Select number of items for the heatmap
    n_heatmap = st.number_input(
        "Select number of items to include in heatmap (max 500 recommended):",
        min_value=10, max_value=len(text_vectors), value=50, step=10
    )

    if st.checkbox("Show similarity heatmap"):
        with torch.no_grad():
            # Sample the first n_heatmap items (or random sample)
            indices = np.arange(n_heatmap)
            sim_matrix_sample = torch.matmul(text_vectors[indices], corpus_vectors[indices].T).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(sim_matrix_sample, cmap="viridis", ax=ax)
        st.pyplot(fig)


    # Compute Top-1, Top-5, Top-10 accuracy
    scores = torch.matmul(text_vectors, corpus_vectors.T)
    top1 = torch.mean((torch.argmax(scores, dim=1) == torch.arange(scores.shape[0], device=scores.device)).float()).item()
    top5 = torch.mean(torch.tensor([i in torch.topk(scores[i],5).indices.tolist() for i in range(scores.shape[0])], dtype=torch.float32)).item()
    top10 = torch.mean(torch.tensor([i in torch.topk(scores[i],10).indices.tolist() for i in range(scores.shape[0])], dtype=torch.float32)).item()
    st.markdown(f"**Top-1 Accuracy:** {top1:.4f}")
    st.markdown(f"**Top-5 Accuracy:** {top5:.4f}")
    st.markdown(f"**Top-10 Accuracy:** {top10:.4f}")


# --- Tab 4: Filter by MBJ Bandgap ---
with tab4:
    print("Tab 4: Filter by MBJ Bandgap")
    st.header("Filter Dataset by MBJ Bandgap")
    min_bg, max_bg = st.slider("Select MBJ bandgap range:", 0.0, 10.0, (0.0, 5.0))
    k_bandgap = st.slider("Number of results to show:", 1, 10, 5)

    # Filter indices
    filtered_idx = [i for i, bg in enumerate(bandgaps) if bg is not None and min_bg <= bg <= max_bg]

    if not filtered_idx:
        st.warning("No items found in this bandgap range.")
    else:
        st.subheader(f"Showing top {min(k_bandgap, len(filtered_idx))} results within selected bandgap")
        for i in filtered_idx[:k_bandgap]:
            img = list_to_image(data_test.iloc[i]["image"])
            img = (img - img.min()) / (img.max() - img.min())
            st.image(img, caption=f"Formula: {formulas[i]}, Bandgap: {bandgaps[i]:.3f}")  # Show bandgap value

        # Download filtered dataset
        filtered_data = data_test.iloc[filtered_idx]
        csv = filtered_data.to_csv(index=False)
        st.download_button("Download Filtered Results", csv, file_name="filtered_bandgap_results.csv")