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

def list_to_image(img_list, size=224):
    """Convert a JSON list to a 2D image."""
    return np.array(json.loads(img_list)).reshape(size, size)

import re
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

MODEL_PATHS = {
    'CLIPP-Allenai': '../models/CLIPP_allenai/checkpoints/best_clipp.pth',
    'CLIPP-BERT': '../models/CLIPP_bert/checkpoints/best_clipp_bert.pth',
    'Salesforce': '../models/Salesforce/checkpoints_blip/best_blip.pth',
    'Apple': '../models/Apple_MobileCLIP/checkpoints/best_clipp_apple.pth'
}

# Load model
@st.cache_resource
def load_selected_model(model_name):
    """Load the selected model and its processor/tokenizer."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")
    
    checkpoint_path = MODEL_PATHS[model_name]

    if model_name == 'CLIPP-Allenai':
        model, processor, dataset = load_clipp_scibert(checkpoint_path, device)
    elif model_name == 'CLIPP-BERT':
        model, processor, dataset = load_clipp_distilbert(checkpoint_path, device)
    elif model_name == 'Apple':
        model, processor, dataset = load_mobileclip(checkpoint_path, device)
    elif model_name == 'Salesforce':
        model, processor, dataset = load_blip(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, processor, dataset

# Load BLIP embeddings from saved pickle file
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- Load dataset ---
@st.cache_resource
def load_dataset(model_name):
    # Define embeddings path based on model_name
    embeddings_dir = Path('./embeddings')
    if model_name == "Apple":
        embeddings_path = embeddings_dir / "val_df_with_embeddings_apple.pkl"
    elif model_name == "Salesforce":
        embeddings_path = embeddings_dir / "val_df_with_embeddings_blip.pkl"
    elif model_name == "CLIPP-Allenai":
        embeddings_path = embeddings_dir / "val_df_with_embeddings_scibert.pkl"
    elif model_name == "CLIPP-BERT":
        embeddings_path = embeddings_dir / "val_df_with_embeddings_distilbert.pkl"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print(f"🔄 Loading embeddings from: {embeddings_path}")

    if embeddings_path.exists():
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    return embeddings_data

# Initialize with default model
if 'model_name' not in st.session_state:
    st.session_state.model_name = "Apple"
    st.session_state.model, st.session_state.processor, st.session_state.dataset = load_selected_model(st.session_state.model_name)


@st.cache_resource
def load_features(_model_name):
    """Load precomputed features for the selected model."""
    df = load_dataset(_model_name)
    
    # Use torch.stack to properly combine embeddings into tensors
    corpus_vectors = torch.stack([torch.tensor(emb) for emb in df['val_img_embs']]).squeeze().to(device)
    text_vectors = torch.stack([torch.tensor(emb) for emb in df['val_txt_embs']]).squeeze().to(device)

    corpus_ids = df['id']
    # Extract formulas and bandgaps
    formulas = []
    bandgaps = []
    for text in df['input']:
        formula_match = re.search(r'The chemical formula is ([A-Za-z0-9]+)', text)
        bandgap_match = re.search(r'mbj_bandgap value is ([0-9.]+)', text)
        formula = formula_match.group(1) if formula_match else "N/A"
        bandgap_str = bandgap_match.group(1) if bandgap_match else None
        if bandgap_str:
            try:
                bandgap = float(bandgap_str.strip().rstrip('.'))
            except Exception:
                bandgap = None
        else:
            bandgap = None
        formulas.append(formula)
        bandgaps.append(bandgap)
    
    return corpus_vectors, text_vectors, corpus_ids, formulas, bandgaps, df


# Streamlit UI
st.title("🔍 Materials Science Retrieval App")

# Model selection
model_name = st.selectbox(
    "Select Model",
    ["CLIPP-Allenai", "CLIPP-BERT", "Apple", "Salesforce"],
    index=["CLIPP-Allenai", "CLIPP-BERT", "Apple", "Salesforce"].index(st.session_state.model_name),
    help="Choose the model to use for retrieval"
)

# Update session state when model_name changes
if model_name != st.session_state.model_name:
    st.session_state.model_name = model_name
    # Clear cached data for the new model
    load_features.clear()
    load_dataset.clear()
    st.session_state.model, st.session_state.processor, st.session_state.dataset = load_selected_model(model_name)
    st.rerun()  # Rerun to refresh the app with the new model

# Load features for current model
corpus_vectors, text_vectors, corpus_ids, formulas, bandgaps, df = load_features(st.session_state.model_name)

# Load selected model
try:
    model, processor, dataset = st.session_state.model, st.session_state.processor, st.session_state.dataset
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
        value="The chemical formula is LiGeS. The mbj_bandgap value is 0.0."
    )
    k = st.slider("Number of top results to show:", 1, 10, 3)

    if st.button("Search Text"):
        st.write("Searching...")
        print("Searching for text...")
        with torch.no_grad():
            if model_name == 'Apple':
                caption, text_tokens = dataset.prepare_caption(text_query)
                embeddings = model.get_text_features(text_tokens.to(device))
            elif model_name == 'Salesforce':
                caption, input_ids, attention_mask = dataset.prepare_caption(text_query)
                embeddings = model.get_text_features(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            elif model_name == "CLIPP-Allenai" or model_name == "CLIPP-BERT":
                caption, input_ids, attention_mask = dataset.prepare_caption(text_query)
                embeddings = model.get_text_features(input_ids.view(1,-1).to(device), attention_mask.view(1,-1).to(device))

            # Search over the full dataset (no bandgap filter)
            sims_img = F.cosine_similarity(embeddings, corpus_vectors)
            sims_txt = F.cosine_similarity(embeddings, text_vectors)
            topk_img = torch.topk(sims_img, k)
            topk_txt = torch.topk(sims_txt, k)

            # Display results in columns
            cols = st.columns(3)

            with cols[0]:
                st.subheader("Top Matching Texts")
                for i in topk_txt.indices:
                    idx = i.item()
                    st.markdown(f"**{df.iloc[idx]['input']}**  \nScore: {sims_txt[i]:.3f}")

            with cols[1]:
                st.subheader("Top Matching Images")
                for i in topk_img.indices:
                    idx = i.item()
                    img = list_to_image(df.iloc[idx]["image"])
                    img = (img - img.min()) / (img.max() - img.min())
                    st.image(
                        img, 
                        caption=f"Formula: {formulas[idx]}, Score: {sims_img[i]:.3f}"
                    )

            with cols[2]:
                st.subheader("Download Results")
                top_results = df.iloc[topk_img.indices.cpu().numpy()]
                csv = top_results.to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="top_results.csv")


with tab2:
    print("Tab 2: Image-to-Text Retrieval")
    st.header("Image-to-Text Retrieval")
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    k_img2txt = st.slider("Top results to show:", 1, 10, 3, key="img2txt_slider")

    if uploaded_image is not None:
        # Display the uploaded image at the top
        st.subheader("Uploaded Image")
        uploaded_pil = Image.open(uploaded_image)
        st.image(uploaded_pil, caption="Your uploaded image", width=300)
        
        img = dataset.prepare_image(uploaded_image).unsqueeze(0)
        with torch.no_grad():
            img_feat = model.get_image_features(img.to(device))  # Warm-up
            
        sims = F.cosine_similarity(img_feat, text_vectors)
        topk = torch.topk(sims, k_img2txt)
        st.subheader("Top matching texts for uploaded image")
        for i in topk.indices:
            st.markdown(f"{df.iloc[i.item()]['input']}  \nScore: {sims[i]:.3f}")
            img_arr = list_to_image(df.iloc[i.item()]["image"])
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
            img = list_to_image(df.iloc[i]["image"])
            img = (img - img.min()) / (img.max() - img.min())
            st.image(img, caption=f"Formula: {formulas[i]}, Bandgap: {bandgaps[i]:.3f}")  # Show bandgap value

        # Download filtered dataset
        filtered_data = df.iloc[filtered_idx]
        csv = filtered_data.to_csv(index=False)
        st.download_button("Download Filtered Results", csv, file_name="filtered_bandgap_results.csv")