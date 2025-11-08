# Full-Page Jazz Leadsheet Recognition - Experiment Guide

This guide walks you through the three experiments to evaluate full-page jazz leadsheet recognition using the SMT architecture.

## Prerequisites

1. **Data preparation** (using `data_prep.py`):
   ```bash
   cd Full_Page_jazzmus

   # Prepare handwritten data with train/val/test splits
   python data_prep.py --dataset_name "PRAIG/JAZZMUS" --output_name "handwritten"

   # Prepare synthetic data (no split)
   python data_prep.py --dataset_name "PRAIG/JAZZMUS" --output_name "synthetic" --synthetic
   ```

   This creates:
   ```
   data/
     handwritten/
       train/images/ (70%)
       train/ground_truth/
       val/images/ (10%)
       val/ground_truth/
       test/images/ (20%)
       test/ground_truth/
     synthetic/
       synthetic/images/
       synthetic/ground_truth/
   ```

2. **System-level checkpoint** from SMT training on Grandstaff:
   - Should be at: `weights/grandstaff_pretrained.ckpt` or similar
   - This will be loaded with `load_pretrained=True`

## The Three Experiments

### Experiment 1: Baseline - System-Level Checkpoint on Full-Page

**Goal:** Establish a baseline by testing a system-level trained model on full-page images (expect horrible results).

**Why poor results?**
- System-level model trained on individual staff systems
- Full-page has multiple staff systems stacked vertically
- Input size mismatch
- Sequence structure incompatible
- This demonstrates why full-page training is necessary

**Command:**
```bash
python train_full_page.py \
  --config config/full_page_baseline.gin \
  --load_pretrained True \
  --freeze_encoder False \
  --epochs 10 \
  --batch_size 1 \
  --patience 5 \
  --output_name "baseline_system_level"
```

**Expected results:**
- CER: 50-80% (very bad)
- SER: 80-100% (very bad)
- Convergence: Poor/no convergence
- This confirms the task requires full-page training

**Config file** (`config/full_page_baseline.gin`):
```gin
FullPageDataModule.data_path = "data/handwritten"
FullPageDataModule.vocab_name = "full_page_vocab"
FullPageDataModule.batch_size = 1
FullPageDataModule.num_workers = 4
FullPageDataModule.fixed_img_height = 256
FullPageDataModule.max_fix_img_width = 2048
```

---

### Experiment 2: Train from Scratch (No Pretrained Weights)

**Goal:** Train SMT model for full-page recognition from scratch, learning all features from jazz data.

**Why this approach?**
- No dependency on pretrained weights
- Model learns jazz-specific patterns
- Establishes performance ceiling for training from scratch
- Useful baseline for comparison with transfer learning

**Command:**
```bash
python train_full_page.py \
  --config config/full_page_no_pretrained.gin \
  --load_pretrained False \
  --epochs 200 \
  --batch_size 2 \
  --accumulate_grad_batches 2 \
  --lr 1e-4 \
  --patience 15 \
  --output_name "from_scratch"
```

**Expected results:**
- CER: 30-50% (reasonable)
- SER: 60-80% (reasonable)
- Training time: 12-24 hours on RTX 3080+
- Gradual improvement throughout training
- Checkpoint saved at: `weights/from_scratch/`

**Key features:**
- Starts from random initialization
- No encoder pretaining
- Learns encoder + decoder jointly
- Takes longer but learns jazz-specific features

**Config file** (`config/full_page_no_pretrained.gin`):
```gin
FullPageDataModule.data_path = "data/handwritten"
FullPageDataModule.vocab_name = "full_page_vocab"
FullPageDataModule.batch_size = 2
FullPageDataModule.num_workers = 4
FullPageDataModule.fixed_img_height = 256
FullPageDataModule.max_fix_img_width = 2048
```

---

### Experiment 3: Train with Pretrained Weights (Transfer Learning)

**Goal:** Use pretrained encoder from system-level model, fine-tune for full-page jazz recognition.

**Why better approach?**
- Encoder learns universal visual features (staff lines, note heads, etc.)
- Only decoder needs to learn jazz-specific sequences
- Faster convergence (encoder already knows music notation)
- Better generalization with less data
- Recommended for practical deployment

**Command Option A: Freeze encoder (recommended for limited data):**
```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder True \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --patience 10 \
  --output_name "pretrained_frozen"
```

**Command Option B: Fine-tune encoder (requires more data/epochs):**
```bash
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder False \
  --epochs 150 \
  --batch_size 2 \
  --lr 5e-5 \
  --patience 12 \
  --output_name "pretrained_finetuned"
```

**Expected results:**
- CER: 15-30% (good)
- SER: 40-60% (good)
- Training time: 6-12 hours on RTX 3080+
- Faster convergence than from-scratch
- Better results than from-scratch (if encoder is good)

**Key differences:**
- **Freeze encoder:** Fast training, lower risk of overfitting, good for limited data
- **Fine-tune encoder:** Longer training, better performance if sufficient data, captures domain-specific visual features

**Config file** (`config/full_page_pretrained.gin`):
```gin
FullPageDataModule.data_path = "data/handwritten"
FullPageDataModule.vocab_name = "full_page_vocab"
FullPageDataModule.batch_size = 4
FullPageDataModule.num_workers = 4
FullPageDataModule.fixed_img_height = 256
FullPageDataModule.max_fix_img_width = 2048
```

---

## Running the Experiments

### Sequential execution (recommended for first run):

```bash
# Experiment 1: Baseline (quick sanity check)
echo "=== Experiment 1: Baseline (System-Level Checkpoint) ==="
python train_full_page.py \
  --config config/full_page_baseline.gin \
  --load_pretrained True \
  --epochs 10 \
  --output_name "exp1_baseline"

# Experiment 2: From Scratch (long training)
echo "=== Experiment 2: From Scratch (No Pretrained) ==="
python train_full_page.py \
  --config config/full_page_no_pretrained.gin \
  --load_pretrained False \
  --epochs 200 \
  --output_name "exp2_from_scratch"

# Experiment 3: With Pretrained Weights
echo "=== Experiment 3: With Pretrained (Frozen) ==="
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder True \
  --epochs 100 \
  --output_name "exp3_pretrained_frozen"

# Optional: Fine-tune variant
echo "=== Experiment 3B: With Pretrained (Fine-Tuned) ==="
python train_full_page.py \
  --config config/full_page_pretrained.gin \
  --load_pretrained True \
  --freeze_encoder False \
  --epochs 150 \
  --output_name "exp3_pretrained_finetuned"
```

### Parallel execution (if multiple GPUs available):

```bash
# Run experiments in background on different GPUs
CUDA_VISIBLE_DEVICES=0 python train_full_page.py --config config/full_page_no_pretrained.gin --output_name "exp2" &
CUDA_VISIBLE_DEVICES=1 python train_full_page.py --config config/full_page_pretrained.gin --output_name "exp3" --load_pretrained True --freeze_encoder True &
wait
```

---

## Comparing Results

After all experiments complete, compare metrics:

```bash
# Results will be in:
weights/exp1_baseline/            # System-level checkpoint on full-page
weights/exp2_from_scratch/        # From-scratch training
weights/exp3_pretrained_frozen/   # Pretrained with frozen encoder
weights/exp3_pretrained_finetuned/ # Pretrained with fine-tuned encoder
```

**Comparison table to fill in:**

| Experiment | CER | SER | Train Time | Convergence | Notes |
|-----------|-----|-----|-----------|-------------|-------|
| Baseline (system-level) | ? | ? | ~1-2h | Bad/No | Expected to be horrible |
| From Scratch | ? | ? | ~12-24h | Good | Learns everything |
| Pretrained (Frozen) | ? | ? | ~6-10h | Very Good | Fast convergence |
| Pretrained (Fine-tuned) | ? | ? | ~10-15h | Excellent | Best results likely |

---

## Inference on New Images

Once trained, run inference on new full-page images:

```bash
# Use best checkpoint from any experiment
python inference.py \
  --checkpoint_path "weights/exp3_pretrained_frozen/best.ckpt" \
  --image_path "path/to/full_page.jpg" \
  --output_path "predictions/pred.txt"
```

The inference pipeline will:
1. Load the trained model
2. Preprocess the image (resize, normalize)
3. Generate predictions token-by-token
4. Decode to **kern format
5. Save/display results

---

## Next Steps: Curriculum Learning

After baseline experiments, implement curriculum learning:

**Phase 1:** Train on synthetic data (1-2 systems) using best model from Exp 3
```bash
python train_curriculum.py \
  --stage 1 \
  --data_path data/synthetic \
  --checkpoint weights/exp3_pretrained_frozen/best.ckpt
```

**Phase 2:** Progressive stages (1-3, 1-4 systems)
**Phase 3:** Mixed real + synthetic data
**Phase 4:** Full handwritten dataset fine-tuning

This curriculum approach typically achieves:
- CER: 10-20% (excellent)
- SER: 30-50% (excellent)

---

## Troubleshooting

**Out of memory:**
- Reduce `batch_size` in config (e.g., 1 or 2)
- Reduce `max_fix_img_width` in config
- Use `accumulate_grad_batches` to simulate larger batches

**Poor convergence:**
- Check learning rate (`--lr`)
- Increase `patience` for early stopping
- Verify data is loaded correctly with `--debug True`

**GPU not being used:**
- Check `nvidia-smi` to verify GPU is free
- Ensure PyTorch CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`

**Model not improving:**
- Verify checkpoint is being loaded: check output for "Loading pretrained weights"
- Check learning rate is appropriate (start with 1e-4)
- Verify data paths are correct

---

## Key Parameters Explained

| Parameter | Impact | Range | Notes |
|-----------|--------|-------|-------|
| `epochs` | Training duration | 10-200 | Higher = longer training, better convergence |
| `batch_size` | Memory usage & gradient quality | 1-8 | Higher = faster training but more memory |
| `lr` | Training speed & stability | 1e-5 to 1e-3 | Lower = safer, slower; higher = faster, riskier |
| `freeze_encoder` | Flexibility vs overfitting | True/False | True = stable, fast; False = more flexible |
| `patience` | Early stopping | 5-20 | Higher = continue training longer, find better convergence |

---

## Architecture Overview

The SMT model for full-page jazz leadsheet recognition:

```
Full-Page Image (variable size)
    ↓
[Encoder] - Vision transformer extracting visual features
    ↓
[Decoder] - Sequence-to-sequence generating **kern tokens
    ↓
**kern format (music notation with chord symbols)
```

**Key insight:** Encoder learns "what music looks like" (universal), Decoder learns "how to transcribe jazz" (jazz-specific).

---

## References

- **SMT Paper:** [Reference]
- **JAZZMUS Dataset:** PRAIG/JAZZMUS on HuggingFace
- **Curriculum Learning:** [Vision-based approach for progressive complexity]
- **Full-page OCR:** [Music document analysis]
