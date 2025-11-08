# Full-Page Jazz Leadsheet Recognition - Architecture Overview

## System Overview

The full-page jazz leadsheet recognition system consists of several integrated components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Full-Page Image                      │
│                    (variable size, grayscale)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │  Image Preprocessing │
                    │  - Resize (preserve ratio)
                    │  - Normalize
                    │  - Pad to batch max
                    └────────────┬───────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   SMT-Trainer Model        │
                    │  ┌──────────────────────┐  │
                    │  │    Encoder           │  │
                    │  │  (Vision Transformer)│  │
                    │  │  - Extract features  │  │
                    │  │  - Learn visual repr.│  │
                    │  └──────────┬───────────┘  │
                    │             │              │
                    │  ┌──────────▼───────────┐  │
                    │  │    Decoder           │  │
                    │  │  (Seq-to-Seq)        │  │
                    │  │  - Generate tokens   │  │
                    │  │  - Jazz-specific     │  │
                    │  └──────────┬───────────┘  │
                    └─────────────┼──────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │   Token Decoding         │
                    │  - Convert tokens to     │
                    │    **kern format         │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │  OUTPUT: **kern Format   │
                    │  (music notation +       │
                    │   chord symbols)         │
                    └──────────────────────────┘
```

## Component Details

### 1. Data Module (`jazzmus/dataset/full_page_datamodule.py`)

**Purpose:** Load and preprocess full-page images for training/inference.

**Key Features:**
- Loads full pages without cropping (unlike system-level which crops individual staffs)
- Dynamic padding to batch max dimensions while preserving aspect ratio
- Automatic vocabulary building from **kern transcriptions
- Support for train/val/test splits
- Teacher forcing for training

**Data Flow:**
```
Raw Images + Annotations
    │
    ├─→ Load image (PIL)
    ├─→ Load annotation (json)
    ├─→ Tokenize **kern to token sequence
    │
    ▼
FullPageGrandStaff Dataset
    │
    ├─→ Get max height/width for batch
    ├─→ Pad images dynamically
    ├─→ Create decoder input (shifted tokens)
    │
    ▼
DataLoader → Batches (image, decoder_input, ground_truth, paths)
```

**Key Methods:**
```python
def load_full_page_set(dataset_dir, split, reduce_ratio, ...):
    # Load images and annotations from split directories

def batch_preparation_full_page(data):
    # Pad images dynamically and prepare sequences

class FullPageDataModule(LightningDataModule):
    # Handles train/val/test splits
    # Manages vocabulary
    # Provides dataloaders
```

**Differences from System-Level:**
| Aspect | System-Level | Full-Page |
|--------|--------------|-----------|
| Input | Individual staff | Full page (multiple staffs) |
| Size | ~256×400 | Variable, often 256×2048+ |
| Processing | Crop to bbox | No cropping, preserve full page |
| Sequence | Single staff | Multiple staffs stacked |

---

### 2. Training Script (`train_full_page.py`)

**Purpose:** Main training entry point for all three experiments.

**Features:**
- Load pretrained weights (optional)
- Freeze/unfreeze encoder (optional)
- Gin configuration support
- Lightning trainer with callbacks (early stopping, checkpointing)
- Weights & Biases logging (optional)

**Training Pipeline:**
```python
# 1. Parse configuration
gin.parse_config_file(config)

# 2. Create datamodule
datamodule = FullPageDataModule(...)
datamodule.setup()

# 3. Create model
model = SMT_Trainer(
    load_pretrained=load_pretrained,
    freeze_encoder=freeze_encoder,
    ...
)

# 4. Optional: Freeze encoder
if freeze_encoder and load_pretrained:
    for param in model.model.encoder.parameters():
        param.requires_grad = False

# 5. Create Lightning trainer
trainer = Trainer(
    callbacks=[EarlyStopping, ModelCheckpoint, LRMonitor],
    ...
)

# 6. Train and evaluate
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)
```

**Key Parameters:**
- `load_pretrained`: Load system-level checkpoint
- `freeze_encoder`: Freeze encoder for transfer learning
- `batch_size`: Batch size (usually 1-4)
- `epochs`: Number of epochs
- `lr`: Learning rate
- `patience`: Early stopping patience

---

### 3. Inference Pipeline (`inference.py`)

**Purpose:** Generate predictions on new full-page images.

**Pipeline:**
```
Image Path
    │
    ▼
Load Model Checkpoint
    │
    ├─→ Model type: SMT_Trainer
    ├─→ Load weights
    ├─→ Set to eval mode
    │
    ▼
Preprocess Image
    │
    ├─→ Read from file (cv2)
    ├─→ Resize preserving aspect ratio
    ├─→ Normalize to [0, 1]
    ├─→ Pad to (1024, 2048)
    │
    ▼
Forward Pass (no_grad)
    │
    ├─→ Encoder: Extract features
    ├─→ Decoder: Generate tokens (autoregressive)
    │
    ▼
Decode Tokens
    │
    ├─→ Convert token IDs to strings
    ├─→ Untokenize to **kern format
    │
    ▼
Save/Display Result
```

**Key Methods:**
```python
class FullPageInference:
    def __init__(checkpoint_path, device="cuda"):
        # Load trained model

    def preprocess_image(image_path, max_height, max_width):
        # Resize, normalize, pad image

    def predict(image_path, return_probs=False):
        # Forward pass and decode

    def save_result(result, output_path):
        # Save prediction to file
```

---

### 4. Configuration Files (Gin)

**Purpose:** Manage experiment configurations without code changes.

**Files:**
- `config/full_page_baseline.gin` - System-level checkpoint on full-page
- `config/full_page_no_pretrained.gin` - Training from scratch
- `config/full_page_pretrained.gin` - Training with pretrained weights
- `config/full_page_curriculum.gin` - Curriculum learning (future)

**Example:**
```gin
# Data loading
FullPageDataModule.data_path = "data/handwritten"
FullPageDataModule.vocab_name = "full_page_vocab"
FullPageDataModule.batch_size = 2
FullPageDataModule.num_workers = 4
FullPageDataModule.fixed_img_height = 256
FullPageDataModule.max_fix_img_width = 2048
```

---

## Transfer Learning Strategy

### Why Transfer Learning Works for This Task

1. **Encoder learns universal visual features:**
   - Staff lines recognition
   - Note head detection
   - Clefs, key signatures, time signatures
   - General music notation understanding

2. **Decoder learns jazz-specific sequences:**
   - **kern token sequence modeling
   - Chord symbols (D:min7, G:7, etc.)
   - Jazz-specific note combinations

3. **Hypothesis:** Visual features are transferable across music types

### Three Experiment Design

**Exp 1: Baseline (System-Level on Full-Page)**
- Load pretrained system-level model
- Keep all weights
- Train on full-page data
- Expected: Horrible results (task mismatch)

**Exp 2: From Scratch**
- Initialize encoder randomly
- Initialize decoder randomly
- Train on full-page data
- Expected: Good results, slow convergence

**Exp 3: Transfer Learning (Recommended)**
- Load pretrained encoder
- Initialize decoder randomly
- Freeze encoder, train decoder only
- Expected: Best results, fast convergence

### Encoder Freezing Strategy

**When to freeze:**
- Limited training data (< 1000 samples)
- Want fast training
- Confident pretrained encoder is good
- Avoid overfitting to small dataset

**When to unfreeze:**
- Abundant training data (> 5000 samples)
- Domain is quite different
- Want best possible performance
- Have sufficient training time

---

## Data Format & Preprocessing

### Input Data Structure
```
data/handwritten/
├── train/
│   ├── images/
│   │   ├── train_0000.jpg
│   │   ├── train_0001.jpg
│   │   └── ...
│   └── ground_truth/
│       ├── train_0000.txt (contains **kern)
│       ├── train_0001.txt
│       └── ...
├── val/
│   ├── images/
│   └── ground_truth/
└── test/
    ├── images/
    └── ground_truth/
```

### **kern Format Example
```
!!!COM: Composer Name
!!!OTL: Piece Title
**kern
*clefG2
*M4/4
*k[f#]
*D:
4D
4E
4F#
4G
=1
2A	2D:min7
2G	2G:7
=2
.	.
*-
```

Key elements:
- `**kern` format for standard notation
- Chord symbols (e.g., `D:min7`) on same line as notes
- Measures marked with `=1`, `=2`, etc.

### Image Preprocessing
```
Original Image (variable size)
    │
    ├─→ Read grayscale (PIL/cv2)
    ├─→ Resize preserving aspect ratio
    │   (fit within 1024×2048)
    ├─→ Normalize to [0, 1]
    ├─→ Convert to tensor
    │
    ▼
Padded Image (1, 1, 1024, 2048)
```

---

## Model Architecture (SMT)

### Encoder
```
Input: (B, 1, H, W) - grayscale image

Vision Transformer:
├─→ Patch embedding
├─→ Positional encoding
├─→ Transformer blocks (attention + feedforward)
└─→ Output: (B, seq_len, d_model) - feature sequence

Purpose: Extract visual features from music notation
```

### Decoder
```
Input: (B, seq_len) - token sequence (teacher forcing)

Sequence-to-Sequence:
├─→ Token embedding
├─→ Positional encoding
├─→ Transformer decoder blocks
│   - Self-attention on input
│   - Cross-attention on encoder output
│   - Feedforward
├─→ Output projection
└─→ Output: (B, seq_len, vocab_size) - logits

Purpose: Generate **kern token sequence
```

### Full Model
```
Encoder Output (visual features)
    │
    ▼
┌─────────────────────────┐
│   Cross-Attention       │
│   (decoder ← encoder)   │
└──────────┬──────────────┘
           │
        Decoder
           │
           ▼
    Logits (vocab_size)
           │
           ▼
    Softmax → Probabilities
           │
           ▼
    Argmax → Token
```

---

## Metrics

### Character Error Rate (CER)
```
CER = (Substitutions + Deletions + Insertions) / Total Characters

Range: 0-100%
- 0% = Perfect
- 50%+ = Poor
- < 30% = Good
```

### System Error Rate (SER)
```
SER = (Systems with errors) / (Total systems)

Range: 0-100%
- 0% = Perfect system recognition
- > 80% = Acceptable
- < 50% = Good
```

---

## Expected Performance

| Experiment | CER | SER | Convergence | Training |
|-----------|-----|-----|-------------|----------|
| Baseline | 50-80% | 80-100% | Poor | ~1-2h |
| From-Scratch | 30-50% | 60-80% | Good | 12-24h |
| Transfer (Frozen) | 15-30% | 40-60% | Very Good | 6-10h |
| Transfer (Fine-tune) | 10-20% | 30-50% | Excellent | 10-15h |

---

## Debugging & Analysis

### Common Issues

1. **Model not improving:**
   - Check learning rate
   - Verify data is loaded correctly
   - Check for data leakage

2. **Out of memory:**
   - Reduce batch size
   - Reduce image dimensions
   - Use gradient accumulation

3. **Diverging loss:**
   - Reduce learning rate
   - Check gradient clipping
   - Verify data normalization

### Analysis Utilities

```python
# Check batch contents
from jazzmus.utils.file_utils import print_smt_batch
print_smt_batch(datamodule.train_dataloader())

# Visualize sample predictions
for batch in datamodule.val_dataloader():
    # Generate predictions
    # Display image + ground truth + prediction
    break

# Calculate metrics
from jazzmus.utils.metrics import calculate_cer, calculate_ser
cer = calculate_cer(ground_truth, prediction)
ser = calculate_ser(ground_truth, prediction)
```

---

## Next Steps: Curriculum Learning

After establishing baseline results, implement curriculum learning:

```
Stage 1: Synthetic Data (1-2 systems)
    ├─→ Train encoder to recognize simple staves
    ├─→ Train decoder to handle short sequences
    └─→ Checkpoint: stage1.ckpt

Stage 2: Synthetic Data (1-3 systems)
    ├─→ Load stage1.ckpt
    ├─→ Increase complexity
    └─→ Checkpoint: stage2.ckpt

Stage 3: Synthetic Data (1-4 systems)
    ├─→ Load stage2.ckpt
    ├─→ Further increase
    └─→ Checkpoint: stage3.ckpt

Fine-tune: Real Data (Mixed systems)
    ├─→ Load stage3.ckpt
    ├─→ Mix synthetic (90%) + handwritten (10%)
    ├─→ Gradually shift to (50% + 50%)
    ├─→ Final: (10% + 90%)
    └─→ Final model
```

**Expected improvement with curriculum:**
- CER: 10-20% (vs 15-30% without)
- SER: 20-40% (vs 40-60% without)

---

## References

- **SMT Architecture:** [Citation]
- **Transfer Learning:** [Vision + Language]
- **Curriculum Learning:** [Progressive complexity]
- **Music OCR:** [Full-page recognition challenges]
- **Jazzmus Dataset:** PRAIG/JAZZMUS (HuggingFace)
