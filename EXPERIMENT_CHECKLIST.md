# Experiment Checklist - Full-Page Jazz Leadsheet Recognition

Use this checklist to track your progress through the three experiments.

---

## Pre-Experiment Setup

- [ ] **Dependencies Installed**
  ```bash
  pip install torch pytorch-lightning gin-config wandb music21
  ```
  - [ ] PyTorch with CUDA support
  - [ ] PyTorch Lightning
  - [ ] Gin-Config
  - [ ] Weights & Biases (optional)

- [ ] **GPU Verification**
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  - [ ] GPU detected
  - [ ] CUDA available

- [ ] **Data Prepared**
  ```bash
  python data_prep.py --output_name "handwritten"
  python data_prep.py --output_name "synthetic" --synthetic
  ```
  - [ ] `data/handwritten/train/` exists with ≥100 images
  - [ ] `data/handwritten/val/` exists
  - [ ] `data/handwritten/test/` exists
  - [ ] `data/synthetic/synthetic/` exists (optional, for curriculum)

- [ ] **Pretrained Checkpoint Available**
  - [ ] System-level checkpoint exists
  - [ ] Path known and verified

---

## Experiment 1: Baseline (System-Level Checkpoint on Full-Page)

**Purpose:** Establish baseline showing poor results with direct transfer

**Expected duration:** 1-2 hours

### Setup
- [ ] Understand expected results: CER 50-80%, SER 80-100%
- [ ] Review EXPERIMENT_GUIDE.md "Experiment 1" section
- [ ] Verify pretrained checkpoint location

### Execution
- [ ] Run command:
  ```bash
  python train_full_page.py \
    --config config/full_page_baseline.gin \
    --load_pretrained True \
    --epochs 10 \
    --output_name "exp1_baseline"
  ```

### Monitoring
- [ ] Watch training progress in console
- [ ] Check loss is decreasing (or stable at high value)
- [ ] Monitor GPU memory usage (`nvidia-smi`)

### Results Capture
- [ ] Final CER value: ____%
- [ ] Final SER value: ____%
- [ ] Convergence behavior: ☐ Good  ☐ Poor  ☐ Diverging
- [ ] Best checkpoint location: `weights/exp1_baseline/`

### Analysis
- [ ] Compare CER/SER to expected (should be high ~60%+)
- [ ] Note observations:
  - ___________________________________________
  - ___________________________________________
  - ___________________________________________

---

## Experiment 2: From Scratch (No Pretrained Weights)

**Purpose:** Train full-page model from scratch as reference

**Expected duration:** 12-24 hours

### Setup
- [ ] Understand expected results: CER 30-50%, SER 60-80%
- [ ] Review EXPERIMENT_GUIDE.md "Experiment 2" section
- [ ] Ensure adequate GPU memory
- [ ] Plan for long training (will run unattended)

### Execution
- [ ] Run command:
  ```bash
  python train_full_page.py \
    --config config/full_page_no_pretrained.gin \
    --load_pretrained False \
    --epochs 200 \
    --batch_size 2 \
    --output_name "exp2_from_scratch"
  ```

### Monitoring
- [ ] Training starts without errors
- [ ] Loss decreases over first 10 epochs (sanity check)
- [ ] Estimated time: ~12-24 hours

### Results Capture
- [ ] Final CER value: ____%
- [ ] Final SER value: ____%
- [ ] Convergence behavior: ☐ Good  ☐ Plateaued  ☐ Still improving
- [ ] Training time: _______ hours
- [ ] Best checkpoint location: `weights/exp2_from_scratch/`
- [ ] Total epochs trained: _______

### Analysis
- [ ] Compare CER/SER to expected (should be moderate ~40%)
- [ ] Compare to Exp 1 (should be much better)
- [ ] Note convergence pattern:
  - ___________________________________________
  - ___________________________________________
  - ___________________________________________

---

## Experiment 3A: Pretrained + Frozen Encoder

**Purpose:** Transfer learning with frozen visual features

**Expected duration:** 6-10 hours

### Setup
- [ ] Understand expected results: CER 15-30%, SER 40-60%
- [ ] Review EXPERIMENT_GUIDE.md "Experiment 3" section
- [ ] Verify pretrained checkpoint loads correctly

### Execution
- [ ] Run command:
  ```bash
  python train_full_page.py \
    --config config/full_page_pretrained.gin \
    --load_pretrained True \
    --freeze_encoder True \
    --epochs 100 \
    --batch_size 4 \
    --output_name "exp3a_pretrained_frozen"
  ```

### Monitoring
- [ ] "Loading pretrained weights" appears in output
- [ ] "Freezing encoder" appears in output
- [ ] Loss drops quickly in first 5 epochs
- [ ] Training is faster than Exp 2

### Results Capture
- [ ] Final CER value: ____%
- [ ] Final SER value: ____%
- [ ] Convergence behavior: ☐ Excellent  ☐ Good  ☐ Slow
- [ ] Training time: _______ hours
- [ ] Best checkpoint location: `weights/exp3a_pretrained_frozen/`
- [ ] Epochs to convergence: _______

### Analysis
- [ ] Compare CER/SER to expected (should be good ~20%)
- [ ] Compare to Exp 2 (should be better than from-scratch)
- [ ] Note speed improvement vs Exp 2
- [ ] Observations:
  - ___________________________________________
  - ___________________________________________
  - ___________________________________________

---

## Experiment 3B: Pretrained + Fine-Tuned Encoder (Optional)

**Purpose:** Transfer learning with fine-tuned visual features

**Expected duration:** 10-15 hours

### Setup
- [ ] Understand expected results: CER 10-20%, SER 30-50%
- [ ] Review EXPERIMENT_GUIDE.md "Experiment 3" section
- [ ] Have sufficient training time

### Execution
- [ ] Run command:
  ```bash
  python train_full_page.py \
    --config config/full_page_pretrained.gin \
    --load_pretrained True \
    --freeze_encoder False \
    --epochs 150 \
    --batch_size 2 \
    --lr 5e-5 \
    --output_name "exp3b_pretrained_finetuned"
  ```

### Monitoring
- [ ] "Loading pretrained weights" appears in output
- [ ] No "Freezing" message (encoder is trainable)
- [ ] Loss decreases steadily
- [ ] Slower than Exp 3A (more parameters to update)

### Results Capture
- [ ] Final CER value: ____%
- [ ] Final SER value: ____%
- [ ] Convergence behavior: ☐ Excellent  ☐ Good  ☐ Slow
- [ ] Training time: _______ hours
- [ ] Best checkpoint location: `weights/exp3b_pretrained_finetuned/`
- [ ] Epochs to convergence: _______

### Analysis
- [ ] Compare CER/SER to expected (should be excellent ~15%)
- [ ] Compare to Exp 3A (should be slightly better)
- [ ] Note whether improvement justifies extra training time
- [ ] Observations:
  - ___________________________________________
  - ___________________________________________
  - ___________________________________________

---

## Results Comparison & Analysis

### Summary Table
| Experiment | CER | SER | Train Time | Notes |
|-----------|-----|-----|-----------|-------|
| 1: Baseline | ____% | ____% | ~1-2h | |
| 2: From-Scratch | ____% | ____% | 12-24h | |
| 3A: Frozen | ____% | ____% | 6-10h | |
| 3B: Fine-Tuned | ____% | ____% | 10-15h | |

### Key Insights
- [ ] Experiment 1 confirms system-level doesn't work directly
  - Observation: ___________________________________________

- [ ] Experiment 2 shows from-scratch baseline
  - Observation: ___________________________________________

- [ ] Experiment 3A shows transfer learning effectiveness
  - Speed improvement vs Exp 2: _____________ x faster
  - Quality improvement: _____________ % better CER

- [ ] Experiment 3B shows fine-tuning benefit (if run)
  - Improvement over 3A: _____________ % CER
  - Worth the extra time: ☐ Yes  ☐ No

### Best Model Selection
- [ ] Recommended model: **Experiment __**
  - Reason: ___________________________________________

- [ ] Checkpoint path: `weights/_________________________/`

---

## Inference Testing

### Test Inference on Validation Set

- [ ] Select best model checkpoint
- [ ] Run inference on sample image:
  ```bash
  python inference.py \
    --checkpoint_path "weights/[BEST_MODEL]/best.ckpt" \
    --image_path "data/handwritten/test/images/test_0000.jpg" \
    --output_path "predictions/test_0000.txt"
  ```

- [ ] Check results:
  - [ ] Prediction file generated
  - [ ] Prediction contains **kern format
  - [ ] Length seems reasonable

- [ ] Manual review:
  - [ ] Compare prediction to ground truth
  - [ ] Are chord symbols recognized?
  - [ ] Are notes correct?

### Batch Inference (Optional)

- [ ] Run inference on 5-10 test samples
- [ ] Calculate metrics:
  - Average CER: ____%
  - Average SER: ____%

- [ ] Visual inspection:
  - [ ] Quality assessment: ☐ Good  ☐ Acceptable  ☐ Poor
  - [ ] Typical errors: ___________________________________________

---

## Documentation & Reporting

- [ ] Create results summary:
  - [ ] File: `results/experiment_summary.txt`
  - [ ] Include: CER/SER for each experiment
  - [ ] Include: Training times
  - [ ] Include: Key insights

- [ ] Save best checkpoints:
  - [ ] Location: `weights/[BEST_MODEL]/`
  - [ ] Backup: `backups/[BEST_MODEL]_backup.ckpt`

- [ ] Document lessons learned:
  - ___________________________________________
  - ___________________________________________
  - ___________________________________________

---

## Next Steps: Curriculum Learning (Future Work)

Once you've completed the baseline experiments:

- [ ] Plan curriculum stages
  - [ ] Stage 1: Synthetic 1-2 systems
  - [ ] Stage 2: Synthetic 1-3 systems
  - [ ] Stage 3: Synthetic 1-4 systems
  - [ ] Fine-tune: Real data

- [ ] Implement curriculum data generator

- [ ] Run curriculum training
  - Expected CER: 5-10%
  - Expected SER: 15-30%

---

## Sign-Off

- [ ] All experiments completed
- [ ] Results documented
- [ ] Best model identified
- [ ] Ready for deployment/further development

**Completion Date:** _______________

**Notes:**
_________________________________________________________________

_________________________________________________________________

_________________________________________________________________
