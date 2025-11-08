# Quick Start - Full-Page Jazz Leadsheet Recognition

## 30-Second Overview

You now have a complete pipeline for full-page jazz leadsheet recognition. The system includes:
1. **Data module** - loads full-page images
2. **Training script** - supports 3 different training approaches
3. **Inference pipeline** - generates predictions
4. **Config files** - experiment configurations
5. **Experiments** - reproducible comparison of 4 approaches

---

## Prerequisites

```bash
# Ensure you have the necessary dependencies
pip install torch pytorch-lightning gin-config wandb music21

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 1: Prepare Data (One-Time Setup)

```bash
cd Full_Page_jazzmus

# Download and organize handwritten data (train/val/test splits)
python data_prep.py --output_name "handwritten"

# Download and organize synthetic data (no splits)
python data_prep.py --output_name "synthetic" --synthetic
```

This creates:
- `data/handwritten/` - with train/val/test splits
- `data/synthetic/` - unsplit synthetic data

---

## Step 2: Run the Three Experiments

### Option A: Automatic (Recommended)

```bash
# Run all experiments with interactive prompts
bash run_experiments.sh all

# Or run specific experiments
bash run_experiments.sh 1      # Baseline only
bash run_experiments.sh 2      # From-scratch only
bash run_experiments.sh 3a     # Pretrained (frozen encoder)
bash run_experiments.sh 3b     # Pretrained (fine-tuned encoder)
```

### Option B: Manual Commands

**Experiment 1: Baseline (System-Level Checkpoint)**
```bash
python train_full_page.py \
  --config config/full_page_baseline.gin \
  --load_pretrained True \
  --epochs 10 \
  --output_name "exp1_baseline"
```
⏱️ Expected time: 1-2 hours | Expected CER: 50-80%

**Experiment 2: From Scratch**
```bash
python train_full_page.py \
  --config config/full_page_no_pretrained.gin \
  --load_pretrained False \
  --epochs 200 \
  --output_name "exp2_from_scratch"
```
⏱️ Expected time: 12-24 hours | Expected CER: 30-50%

**Experiment 3a: Pretrained (Frozen Encoder)**
```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder True \
  --epochs 100 \
  --output_name "exp3a_pretrained_frozen"
```
⏱️ Expected time: 6-10 hours | Expected CER: 15-30%

**Experiment 3b: Pretrained (Fine-Tuned Encoder)**
```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder False \
  --epochs 150 \
  --output_name "exp3b_pretrained_finetuned"
```
⏱️ Expected time: 10-15 hours | Expected CER: 10-20%

---

## Step 3: Run Inference on New Images

```bash
python inference.py \
  --checkpoint_path "weights/exp3a_pretrained_frozen/best.ckpt" \
  --image_path "path/to/full_page.jpg" \
  --output_path "predictions/output.txt"
```

Output: Predictions saved as **kern format music notation

---

## Key Files & Directories

### Code Files
```
train_full_page.py          Main training script
inference.py                Prediction script
data_prep.py               Data download & organization
jazzmus/
  dataset/
    full_page_datamodule.py   Data loading module
```

### Configuration Files
```
config/
  full_page_baseline.gin                  System-level on full-page
  full_page_no_pretrained.gin            From scratch
  full_page_pretrained.gin               With pretrained weights
  full_page_curriculum.gin               Curriculum learning (future)
```

### Documentation
```
README.md                   Main overview
EXPERIMENT_GUIDE.md         Detailed experiment guide
ARCHITECTURE_OVERVIEW.md    Technical architecture
QUICK_START.md             This file
```

### Data
```
data/
  handwritten/
    train/
    val/
    test/
  synthetic/
```

### Results
```
weights/
  exp1_baseline/
  exp2_from_scratch/
  exp3a_pretrained_frozen/
  exp3b_pretrained_finetuned/
```

---

## Common Commands Cheatsheet

```bash
# View help for training
python train_full_page.py --help

# View help for inference
python inference.py --help

# Run in debug mode (fast iteration)
python train_full_page.py --debug True --epochs 2

# Resume training from checkpoint
python train_full_page.py \
  --checkpoint weights/exp3a_pretrained_frozen/epoch-05.ckpt

# Run on CPU (if GPU unavailable)
CUDA_VISIBLE_DEVICES="" python train_full_page.py --config config/full_page_no_pretrained.gin

# Run with different batch size
python train_full_page.py --batch_size 1 --config config/full_page_no_pretrained.gin
```

---

## Expected Results Summary

| Experiment | Description | CER | SER | Time |
|-----------|-------------|-----|-----|------|
| 1 | System-level checkpoint (baseline) | 50-80% | 80-100% | 1-2h |
| 2 | From scratch (no pretrained) | 30-50% | 60-80% | 12-24h |
| 3a | Pretrained + frozen encoder | 15-30% | 40-60% | 6-10h |
| 3b | Pretrained + fine-tuned | 10-20% | 30-50% | 10-15h |

**CER** = Character Error Rate (lower is better)
**SER** = System Error Rate (lower is better)

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_full_page.py --batch_size 1

# Reduce image dimensions in config
# Edit config/full_page_*.gin and change:
# FullPageDataModule.max_fix_img_width = 1024
```

### GPU Not Being Used
```bash
# Check GPU is available
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force specific GPU
CUDA_VISIBLE_DEVICES=0 python train_full_page.py ...
```

### Data Not Loading
```bash
# Verify data exists
ls -la data/handwritten/train/images/
ls -la data/handwritten/train/ground_truth/

# Run with debug to see loading progress
python train_full_page.py --debug True
```

### Model Not Improving
- Increase `--epochs` (e.g., 200 instead of 100)
- Decrease `--lr` (e.g., 5e-5 instead of 1e-4)
- Increase `--patience` (e.g., 20 instead of 10)

---

## Next Steps

### After Initial Experiments
1. Analyze results in `EXPERIMENT_GUIDE.md`
2. Pick best model (likely Exp 3a or 3b)
3. Run inference on validation set
4. Compare predictions with ground truth

### Advanced: Curriculum Learning
Implement multi-stage training with synthetic data:
```
Stage 1: Synthetic (1-2 systems)
    ↓
Stage 2: Synthetic (1-3 systems)
    ↓
Stage 3: Synthetic (1-4 systems)
    ↓
Fine-tune: Real data
```

Expected improvement: CER 10-20% → CER 5-10%

---

## Full Documentation

For detailed information, see:
- **EXPERIMENT_GUIDE.md** - Each experiment explained in detail
- **ARCHITECTURE_OVERVIEW.md** - System architecture and design decisions
- **README.md** - Overview and setup guide

---

## Support

If something doesn't work:

1. Check the error message
2. See "Troubleshooting" section above
3. Review relevant documentation
4. Check EXPERIMENT_GUIDE.md for your specific experiment

Key resources:
- Data issues → See EXPERIMENT_GUIDE.md "Prerequisites"
- Training issues → See ARCHITECTURE_OVERVIEW.md "Debugging"
- Results interpretation → See EXPERIMENT_GUIDE.md "Expected results"
