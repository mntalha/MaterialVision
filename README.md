# MaterialVision: Interactive Materials Science Retrieval System

MaterialVision is an advanced multimodal retrieval system that bridges materials science textual descriptions with STEM (Scanning Transmission Electron Microscopy) imaging data. The project features a web application for interactive text-to-image and image-to-text retrieval using state-of-the-art vision-language models.

## 🎯 Key Features

### 1. **Interactive Web Application**
- **Text-to-Image Retrieval**: Search for STEM images using materials descriptions
- **Image-to-Text Retrieval**: Upload images to find matching textual descriptions  
- **Multi-Model Support**: Compare results across different vision-language models
- **Real-time Metrics**: View retrieval performance and similarity heatmaps
- **Bandgap Filtering**: Filter materials by MBJ bandgap ranges

### 2. **Multiple Model Architectures**
- **CLIPP-SciBERT**: Custom CLIP-style model with scientific text understanding
- **CLIPP-DistilBERT**: Lightweight version with DistilBERT text encoder
- **MobileCLIP**: Apple's efficient mobile-optimized model
- **BLIP**: Salesforce's advanced vision-language model

### 3. **Comprehensive Pipeline**
- Materials property extraction and parsing
- STEM image processing and normalization
- Embedding generation and caching
- Interactive visualization and analysis

## 🚀 Quick Start

### Web Application
```bash
# Navigate to webapp directory
cd webapp

# Run the Streamlit app
streamlit run app.py
```

### Model Loading and Embedding Generation
```python
# Import model functions
from models import load_clipp_scibert, load_mobileclip, load_blip

# Load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'models/CLIPP_allenai/checkpoints/best_clipp.pth'
model, tokenizer, dataset = load_clipp_scibert(checkpoint_path, device)

# Generate text embeddings
text = "The chemical formula is LiGeS. The mbj_bandgap value is 0.0."
tokens = tokenizer(text, return_tensors="pt", max_length=512).to(device)
text_features = model.get_text_features(tokens['input_ids'], tokens['attention_mask'])
```

## 📁 Repository Structure

```
MaterialVision/
├── webapp/
│   ├── app.py              # Main Streamlit web application
│   ├── models.py           # Model loading utilities
│   ├── embeddings/         # Precomputed embeddings cache
│   └── simple_text_embedding.ipynb  # Embedding generation notebook
├── models/
│   ├── CLIPP_allenai/      # SciBERT-based CLIPP model
│   ├── CLIPP_bert/         # DistilBERT-based CLIPP model  
│   ├── Apple_MobileCLIP/   # Apple MobileCLIP model
│   └── Salesforce/         # BLIP model implementation
├── data/
│   ├── alpaca_mbj_bandgap_train.csv    # Training dataset
│   ├── alpaca_mbj_bandgap_test.csv     # Validation dataset
│   └── train/test/         # Image data directories
└── tests/                  # Development notebooks and tests
```

## 💻 Usage Examples

### Web Application Features

#### Text-to-Image Retrieval
Search for STEM images using materials descriptions:
```python
# Example query in the web app
text_query = "The chemical formula is LiGeS. The mbj_bandgap value is 0.0."

# Returns: Top matching images with similarity scores
# - Formula parsing: "1 Li 1 Ge 1 S"  
# - Bandgap extraction: 0.0 eV
# - Visual similarity ranking
```

#### Image-to-Text Retrieval  
Upload STEM images to find matching descriptions:
```python
# Upload any materials image
# Returns: Most similar textual descriptions from dataset
# Shows: Chemical formulas, bandgap values, similarity scores
```

### Programmatic Usage

#### Custom Text Embedding Generation
```python
# Generate embeddings for custom materials descriptions
from models import load_clipp_scibert

def create_text_embeddings(model_name, texts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'CLIPP-SciBERT':
        checkpoint_path = 'models/CLIPP_allenai/checkpoints/best_clipp.pth'
        model, tokenizer, _ = load_clipp_scibert(checkpoint_path, device)
        
        embeddings = []
        model.eval()
        with torch.no_grad():
            for text in texts:
                tokens = tokenizer(text, padding=True, truncation=True, 
                                 return_tensors="pt", max_length=512).to(device)
                text_features = model.get_text_features(
                    tokens['input_ids'], tokens['attention_mask']
                )
                embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(embeddings)

# Usage
texts = ["Silicon carbide semiconductor", "Iron oxide magnetic material"]
embeddings = create_text_embeddings('CLIPP-SciBERT', texts)
```

#### Chemical Formula Processing
```python
import re

def parse_chemical_formula(formula):
    """Convert Fe2O3 -> 2 Fe 3 O format for better model understanding"""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    result_parts = []
    for element, count in matches:
        if not count:
            count = "1"
        result_parts.extend([count, element])
    
    return ' '.join(result_parts)

# Example
formula = "Fe2O3"
parsed = parse_chemical_formula(formula)  # Returns: "2 Fe 3 O"
```

#### Similarity Search with Filtering
```python
import torch.nn.functional as F

# Search with bandgap filtering
def search_by_bandgap_range(text_query, min_bandgap, max_bandgap, k=5):
    # Generate query embedding
    query_embedding = model.get_text_features(...)
    
    # Filter by bandgap range
    filtered_indices = [
        i for i, bg in enumerate(bandgaps) 
        if bg is not None and min_bandgap <= bg <= max_bandgap
    ]
    
    # Compute similarities for filtered set
    filtered_vectors = corpus_vectors[filtered_indices]
    similarities = F.cosine_similarity(query_embedding, filtered_vectors)
    
    # Get top-k results
    topk = torch.topk(similarities, k)
    return filtered_indices[topk.indices], topk.values

# Usage
results, scores = search_by_bandgap_range(
    "semiconductor material", min_bandgap=1.0, max_bandgap=3.0, k=5
)
```

## 📊 Model Performance & Comparison

### Available Models

| Model | Text Encoder | Vision Encoder | Features |
|-------|--------------|----------------|----------|
| **CLIPP-SciBERT** | SciBERT (scientific vocabulary) | ViT-Base/16 | Best for scientific text |
| **CLIPP-DistilBERT** | DistilBERT (lightweight) | ViT-Base/16 | Faster inference |
| **MobileCLIP** | MobileBERT | MobileViT | Mobile-optimized |
| **BLIP** | BERT | ViT-Large/16 | General-purpose VLM |

### Retrieval Performance (Top-k Accuracy)

#### Validation Set Results
```
CLIPP-SciBERT:
├── Text→Image: Top-1: 36.9%  Top-5: 65.1%  Top-10: 74.9%
└── Image→Text: Top-1: 36.6%  Top-5: 66.2%  Top-10: 74.9%

CLIPP-DistilBERT: 
├── Text→Image: Top-1: 12.5%  Top-5: 36.6%  Top-10: 49.8%
└── Image→Text: Top-1: 14.2%  Top-5: 37.4%  Top-10: 50.6%

Apple MobileCLIP:
├── Text→Image: Top-1: 38.0%  Top-5: 67.0%  Top-10: 76.7%
└── Image→Text: Top-1: 35.9%  Top-5: 65.4%  Top-10: 77.6%

BLIP (Salesforce):
├── Text→Image: Top-1: 46.8%  Top-5: 72.9%  Top-10: 80.9%
└── Image→Text: Top-1: 45.3%  Top-5: 73.6%  Top-10: 80.1%
```

#### Training Set Results  
```
CLIPP-SciBERT:
├── Text→Image: Top-1: 44.9%  Top-5: 80.5%  Top-10: 90.3%
└── Image→Text: Top-1: 47.2%  Top-5: 81.9%  Top-10: 90.9%

CLIPP-DistilBERT:
├── Text→Image: Top-1: 14.6%  Top-5: 39.3%  Top-10: 52.6%
└── Image→Text: Top-1: 14.3%  Top-5: 40.3%  Top-10: 54.5%

Apple MobileCLIP:
├── Text→Image: Top-1: 63.0%  Top-5: 93.8%  Top-10: 97.8%
└── Image→Text: Top-1: 60.5%  Top-5: 92.4%  Top-10: 97.4%

BLIP (Salesforce):
├── Text→Image: Top-1: 57.1%  Top-5: 90.5%  Top-10: 96.9%
└── Image→Text: Top-1: 56.6%  Top-5: 90.4%  Top-10: 96.4%
```

### Model Architecture Details

#### CLIPP Models
```python
# CLIP-style contrastive learning architecture
class CLIPPModel:
    def __init__(self):
        self.vision_encoder = ViT_Base_16()      # 224x224 patches
        self.text_encoder = SciBERT()            # Scientific vocabulary
        self.projection_dim = 256                # Joint embedding space
        self.temperature = 0.07                  # Contrastive learning temp
    
    def forward(self, images, texts):
        img_features = self.vision_encoder(images)
        txt_features = self.text_encoder(texts)
        
        # Project to shared space
        img_embeds = self.img_projection(img_features)
        txt_embeds = self.txt_projection(txt_features)
        
        return F.normalize(img_embeds), F.normalize(txt_embeds)
```

#### BLIP Model  
```python
# Advanced vision-language model with cross-modal attention
class BLIPModel:
    def __init__(self):
        self.vision_encoder = ViT_Large_16()     # Larger vision model
        self.text_encoder = BERT()               # Standard BERT
        self.cross_attention = MultiheadAttention() # Cross-modal fusion
        
    def forward(self, images, texts):
        # Dual-encoder + cross-attention architecture
        vision_embeds = self.vision_encoder(images)
        text_embeds = self.text_encoder(texts)
        
        # Cross-modal attention for better alignment
        fused_embeds = self.cross_attention(vision_embeds, text_embeds)
        return fused_embeds
```

## 🔬 Applications & Use Cases

### Materials Discovery via Web Interface
The Streamlit web application (`webapp/app.py`) provides interactive search capabilities:

- **Text-to-Image Search**: Enter materials descriptions to find similar STEM images
- **Image Upload**: Upload STEM images to find matching textual descriptions  
- **Model Comparison**: Switch between different vision-language models
- **Real-time Results**: Get similarity scores and rankings instantly

### Bandgap Range Filtering
The web interface includes a dedicated "Filter by Bandgap" tab:

- **Interactive Sliders**: Adjust bandgap range (0.0 - 10.0 eV)
- **Visual Results**: View filtered STEM images with chemical formulas
- **Download Options**: Export filtered datasets as CSV files
- **Material Classification**: Explore metals, semiconductors, and insulators

### Programmatic Analysis
Use the Jupyter notebooks for custom analysis:

- **`simple_text_embedding.ipynb`**: Generate embeddings for custom materials
- **Model Evaluation Notebooks**: Compare performance across different models
- **Chemical Formula Parsing**: Built-in functions convert formulas to model-friendly format

### Performance Monitoring
The web app provides real-time metrics:

- **Similarity Heatmaps**: Visualize embedding relationships
- **Top-k Accuracy Display**: Monitor retrieval performance
- **Model Comparison**: Side-by-side performance analysis

## 🎮 Web Interface Features

### Interactive Search
- **Real-time text search** with instant similarity scoring
- **Drag-and-drop image upload** for reverse image search
- **Dynamic model switching** to compare different approaches
- **Adjustable result count** (Top-1 to Top-10)

### Visualization & Analysis  
- **Similarity heatmaps** for understanding model behavior
- **Performance metrics** with Top-k accuracy display
- **Chemical formula parsing** with element separation
- **Bandgap range filtering** for targeted material discovery
- **t-SNE embedding visualization** for exploring multimodal alignment

## 📊 Embedding Visualization

### t-SNE Analysis
Each model evaluation notebook includes t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization to analyze how well the models align image and text embeddings in a shared space:

- **Dimensionality Reduction**: Projects high-dimensional embeddings to 2D for visualization
- **Multimodal Alignment**: Shows how closely paired text-image embeddings cluster together
- **Model Comparison**: Visual assessment of different model architectures' alignment quality
- **Pair Highlighting**: First 15 text-image pairs are connected with lines to show correspondence

#### Generated Visualizations:
- `clipp_scibert_tsne.png` - CLIPP-SciBERT embedding space
- `clipp_distilbert_tsne.png` - CLIPP-DistilBERT embedding space  
- `mobileclip_apple_tsne.png` - Apple MobileCLIP embedding space
- `salesforce_blip_tsne.png` - Salesforce BLIP embedding space

Good multimodal alignment is indicated by:
- **Close clustering** of corresponding text-image pairs
- **Short connection lines** between matched embeddings
- **Distinct separation** from non-matching pairs

### Export & Download
- **CSV export** of search results with metadata
- **Filtered dataset downloads** based on bandgap criteria
- **Embedding vectors** for further analysis

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install torch torchvision transformers
pip install streamlit pandas numpy pillow
pip install open-clip-torch  # For MobileCLIP
pip install transformers[torch]  # For BLIP/SciBERT
```

### Quick Setup
```bash
# Clone repository
git clone https://github.com/your-username/MaterialVision.git
cd MaterialVision

# Download model checkpoints (place in respective model directories)
# Run web application
cd webapp && streamlit run app.py
```

## 🔧 Development & Customization

### Adding New Models
```python
# Extend models.py with your custom loader
def load_custom_model(checkpoint_path, device):
    model = YourCustomModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer, dataset

# Add to app.py model selection
MODEL_PATHS['YourModel'] = 'path/to/checkpoint.pth'
```

### Custom Embedding Generation
```python
# Generate embeddings for new datasets
def process_new_dataset(data_path, model_name):
    df = pd.read_csv(data_path)
    model, tokenizer, _ = load_model(model_name)
    
    embeddings = []
    for text in df['descriptions']:
        embedding = generate_embedding(model, tokenizer, text)
        embeddings.append(embedding)
    
    df['embeddings'] = embeddings
    return df
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. Areas for improvement:

- **New model architectures** for better materials understanding
- **Dataset expansion** with additional materials properties  
- **Performance optimizations** for faster inference
- **UI/UX improvements** for the web application
- **Documentation and tutorials** for new users

### Development Workflow
```bash
# Fork the repository
git fork https://github.com/original/MaterialVision.git

# Create feature branch  
git checkout -b feature/your-improvement

# Make changes and test
pytest tests/  # Run tests
streamlit run webapp/app.py  # Test web app

# Submit pull request
git push origin feature/your-improvement
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Issues**: Open a GitHub issue for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for questions and community support  
- **Documentation**: Check the `/tests/` notebooks for usage examples

## 🙏 Acknowledgments

- **Hugging Face Transformers** for model implementations
- **Streamlit** for the web application framework
- **OpenAI CLIP** for the foundational architecture
- **Materials Science Community** for domain expertise and feedback

---

## 📚 Additional Resources

### Notebooks
- [`simple_text_embedding.ipynb`](webapp/simple_text_embedding.ipynb) - Embedding generation pipeline
- [`howtoreadData.ipynb`](tests/howtoreadData.ipynb) - Data loading and analysis
- [`image_test.ipynb`](tests/image_test.ipynb) - STEM image processing examples

### Model Training
- Training logs and loss curves available in each model's checkpoint directory
- Hyperparameter configurations in model-specific config files
- Performance benchmarks on materials science datasets

**🔬 Happy Materials Discovery! 🧪**
