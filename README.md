# Full-Page Jazz Leadsheet Recognition - Training Pipeline

This guide walks through building and training a model for full-page jazz leadsheet OCR using the SMT architecture.

## Overview

We'll:
1. ✅ Create a DataModule for full-page images
2. ✅ Create an inference pipeline
3. ✅ Test system-level checkpoint (expect poor results)
4. ✅ Train SMT with pretrained weights
5. ✅ Train SMT from scratch
6. ✅ Compare results

## Step 1: Data Preparation (DONE)

```bash
# Prepare handwritten data with splits (train/val/test)
python data_prep.py --dataset_name "PRAIG/JAZZMUS" --output_name "handwritten"

# Prepare synthetic data (single folder, no split)
python data_prep.py --dataset_name "PRAIG/JAZZMUS" --output_name "synthetic" --synthetic
```

Output structure:
```
data/
  handwritten/
    train/images/, ground_truth/
    val/images/, ground_truth/
    test/images/, ground_truth/
  synthetic/
    synthetic/images/, ground_truth/
```

## Step 2: DataModule for Full-Page

Create `jazzmus/dataset/full_page_datamodule.py` - loads full-page images without cropping.

```python
# Key differences from system-level:
- Load full-page images as-is
- No bounding box cropping
- Dynamic padding for variable sizes
- Supports curriculum learning (variable # of systems)
```

## Step 3: Inference Pipeline

Create `inference.py` - predict on new full-page images.

```python
# Steps:
1. Load trained model checkpoint
2. Process image (resize, normalize)
3. Generate predictions token-by-token
4. Decode to kern format
5. Save/display results
```

## Step 4: Train with System-Level Checkpoint (Expect Bad Results!)

```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --epochs 10 \
  --batch_size 1
```

**Why bad results?**
- System-level checkpoint trained on individual staff systems
- Full-page has multiple systems vertically stacked
- Model expects different input size and sequence structure
- Transfer learning not directly applicable

## Step 5: Train from Scratch (Best Approach)

```bash
python train_full_page.py \
  --config config/full_page_no_pretrained.gin \
  --load_pretrained False \
  --epochs 200 \
  --batch_size 2
```

## Step 6: Train with Pretrained Weights (Better Approach)

```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder True \
  --epochs 100 \
  --batch_size 2
```

**Why better?**
- Encoder learns visual features (universal across all music)
- Only decoder needs to learn jazz-specific sequences
- Faster convergence
- Better generalization

## File Structure

```
Full_Page_jazzmus/
  data_prep.py                    # Download & organize data
  train_full_page.py              # Main training script
  inference.py                    # Prediction pipeline
  data/
    handwritten/
    synthetic/
  jazzmus/
    dataset/
      full_page_datamodule.py    # NEW: DataModule for full-page
    model/
      full_page_trainer.py        # NEW: Lightning trainer wrapper
    utils/
      inference_utils.py          # NEW: Helper functions
  config/
    full_page_no_pretrained.gin   # NEW: Config without pretraining
    full_page_pretrained.gin      # NEW: Config with pretraining
    full_page_curriculum.gin      # NEW: Config for curriculum learning
```

## Expected Results

### Test 1: System-Level Checkpoint on Full-Page (Baseline - Expect Horrible)
- **CER: 50-80%** (very bad)
- **SER: 80-100%** (very bad)
- **Reason:** Trained on different task (single systems vs full pages)

### Test 2: Train from Scratch
- **CER: 30-50%** (reasonable)
- **SER: 60-80%** (reasonable)
- **Training time: 12-24 hours** (GPU)
- **Why:** Learning everything from scratch takes time

### Test 3: Train with Pretrained Weights
- **CER: 15-30%** (good)
- **SER: 40-60%** (good)
- **Training time: 6-12 hours** (GPU)
- **Why:** Encoder already knows visual features

## Next Steps: Curriculum Learning

After baseline training, implement curriculum:

```python
# Stage 1: Train on synthetic data (1-2 systems)
# Stage 2: Train on synthetic data (1-3 systems)
# Stage 3: Train on synthetic data (1-4 systems)
# Fine-tune: Mix synthetic (90%) + handwritten (10%), gradually shift to (20% + 80%)
```

This will achieve better results than simple fine-tuning.

## Hardware Requirements

- **GPU:** NVIDIA RTX 3080+ or similar (11GB+ VRAM)
- **RAM:** 16GB+ system RAM
- **Storage:** 50GB+ for datasets
- **Training time:** 6-24 hours depending on method

## Quick Start

```bash
# 1. Prepare data
python data_prep.py --synthetic --output_name "synthetic"
python data_prep.py --output_name "handwritten"

# 2. Test system-level (expect bad results)
python train_full_page.py --config config/full_page_baseline.gin --epochs 5

# 3. Train from scratch
python train_full_page.py --config config/full_page_no_pretrained.gin --epochs 200

# 4. Train with pretrained weights
python train_full_page.py --config config/full_page_pretrained.gin --epochs 100

# 5. Run inference
python inference.py --checkpoint weights/best.ckpt --image test.jpg
```
