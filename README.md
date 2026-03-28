# EEG2Text — EEG-to-Language Decoding

A self-contained project for decoding EEG brain signals into natural language descriptions using a frozen CSBrain encoder and a LoRA-fine-tuned TinyLlama language model.

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA GPU (8GB+ VRAM recommended — tested on RTX 4060 Laptop 8GB)
- All dependencies already satisfied if running alongside CSBrain

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `bitsandbytes` requires CUDA. CPU-only is not supported for training.

### 2. Prepare the data

The raw `.mat` files should be in `data/BCICIV2a/raw/`. Run the preprocessing script to bandpass-filter, resample, and write the LMDB database:

```bash
python prepare_data.py --skip_download
```

This creates `data/BCICIV2a/processed_lmdb/` with train (2,784), val (1,152), and test (1,152) splits.

If you don't have the raw files, drop `--skip_download` to download them automatically (~1.4 GB):

```bash
python prepare_data.py
```

Expected output:
```
Processing train split: ['A01', 'A02', 'A03', 'A04', 'A05']
  Processing A01T.mat ... → 288 trials
  ...
  train: 2784 samples
  val:   1152 samples
  test:  1152 samples
LMDB written to: data/BCICIV2a/processed_lmdb
```

### 3. Train the model

```bash
bash sh/finetune_eeg_llm_bcic.sh
```

Or run directly with custom arguments:

```bash
python finetune_eeg_llm.py \
    --downstream_dataset BCICIV2a \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_bcic_new \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --cuda 0
```

Training runs in two phases and takes ~30–60 min on an RTX 4060:

```
Phase 1: Projection warmup (5 epochs)
  Epoch 1 [warmup]: Loss=2.3412, LR=0.001000, Time=4.2min
  Epoch 1 Val Accuracy: 0.2769 (319/1152)
  ...
Phase 2: Joint projection + LoRA training (15 epochs)
  Epoch 6 [joint]: Loss=1.8734, LR=0.000180, Time=5.1min
  Epoch 6 Val Accuracy: 0.3681 (424/1152)  ← best, saving...
  ...
Test Accuracy: 0.3134 (361/1152)
```

Weights are saved to `pth_downtasks/eeg_llm_bcic_new/`:
- `projection_epoch6.pth` — EEGProjection + EEGTokenReducer weights
- `lora_epoch6/` — LoRA adapter (HuggingFace PEFT format)

### 4. Generate text from EEG

```bash
python generate.py \
    --foundation_dir pth/CSBrain.pth \
    --projection_dir pth_downtasks/eeg_llm_bcic_new/projection_epoch6.pth \
    --lora_dir pth_downtasks/eeg_llm_bcic_new/lora_epoch6 \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples 5
```

Expected output:
```
Sample 1:
  True class : 0 — left hand
  Generated  : The EEG recording shows motor imagery patterns consistent with left
               hand movement. Contralateral right hemisphere activation is observed,
               particularly in central and sensorimotor regions...

Sample 2:
  True class : 2 — feet
  Generated  : The EEG shows bilateral central midline activation consistent with
               feet motor imagery. Midline sensorimotor activity with strong CZ
               and CPZ involvement...
```

> **Tip**: Use the best checkpoint epoch number from training (the one logged as "new best") in `--projection_dir` and `--lora_dir`.

---

## Architecture

```
EEG Signal (batch, channels, patches, patch_size)
        │
        ▼
┌─────────────────────────────────────────┐
│  CSBrain Encoder  [FROZEN]              │
│  12-layer transformer, d_model=200      │
│  Cross-scale temporal + region spatial  │
│  attention                              │
└─────────────────────────────────────────┘
        │  (batch, n_channels, n_patches, 200)
        ▼
┌─────────────────────────────────────────┐
│  EEGTokenReducer                        │
│  Pools channels within brain regions   │
│  BCIC: 3 regions × 4 patches = 12 tok  │
│  FACED: 5 regions × patches = ~75 tok  │
└─────────────────────────────────────────┘
        │  (batch, n_tokens, 200)
        ▼
┌─────────────────────────────────────────┐
│  EEGProjection  [TRAINABLE]             │
│  2-layer MLP: 200 → 2048 → 2048        │
└─────────────────────────────────────────┘
        │  (batch, n_tokens, 2048)
        ▼
┌─────────────────────────────────────────┐
│  TinyLlama-1.1B-Chat  [LoRA ADAPTER]   │
│  4-bit NF4 quantization                 │
│  LoRA r=8, α=16 on q_proj & v_proj     │
│  ~1.1M trainable params (0.10%)        │
└─────────────────────────────────────────┘
        │
        ▼
  Natural Language Description
  e.g. "The EEG shows motor imagery patterns
        consistent with left hand movement..."
```

### Brain Region Mapping (BCIC-IV-2a, 22 channels)
| Region | Channels |
|--------|----------|
| Frontal-Central | FC1, FC2, FC3, FC4, FC5, FC6 |
| Central | C1, C2, C3, C4, C5, C6, CZ |
| Central-Parietal | CP1, CP2, CP3, CP4, CP5, CP6, CPZ |

---

## Project Structure

```
EEG2Text/
├── models/
│   ├── CSBrain.py                  # Pretrained EEG foundation encoder
│   ├── CSBrain_transformer.py      # Transformer building blocks
│   ├── CSBrain_transformerlayer.py # Custom transformer layer
│   └── eeg_llm.py                  # EEGLanguageModel, EEGTokenReducer, EEGProjection
├── datasets/
│   ├── bciciv2a_llm_dataset.py     # BCIC-IV-2a motor imagery with text labels + collator
│   ├── bciciv2a_dataset.py         # BCIC-IV-2a classification loader (reference)
│   └── faced_llm_dataset.py        # FACED emotion dataset with text labels
├── utils/
│   ├── signaltools.py              # FFT-based EEG resampling (PyTorch)
│   └── util.py                     # General utilities
├── data/
│   └── BCICIV2a/
│       ├── raw/                    # .mat files (A01T.mat, A01E.mat, ... A09E.mat)
│       └── processed_lmdb/        # LMDB database (train/val/test splits)
├── pth/
│   └── CSBrain.pth                 # Pretrained CSBrain encoder weights
├── pth_downtasks/
│   └── eeg_llm_bcic/
│       ├── best_projection.pth     # Trained EEGProjection weights
│       └── best_lora/              # Trained LoRA adapter (HuggingFace PEFT format)
├── sh/
│   ├── prepare_data.sh             # Data download + preprocessing
│   ├── finetune_eeg_llm_bcic.sh   # Train on BCIC-IV-2a
│   └── finetune_eeg_llm_faced.sh  # Train on FACED
├── prepare_data.py                 # Data download + LMDB preprocessing script
├── finetune_eeg_llm.py             # Training entry point
├── finetune_eeg_llm_trainer.py     # EEGLLMTrainer (2-phase training)
├── generate.py                     # Inference script
├── eeg_llm_notebook.ipynb          # End-to-end notebook (BCIC-IV-2a)
├── EEG_LLM_Architecture.md        # Detailed architecture notes
└── requirements.txt
```

---

---

## Training Strategy

| Phase | Epochs | Trainable | LR |
|-------|--------|-----------|----|
| Warmup | 1–5 | EEGProjection only | 5e-4 |
| Joint | 6–20 | EEGProjection + LoRA | 2e-4 |

- Optimizer: AdamW (weight_decay=0.01)
- LR schedule: Cosine annealing (eta_min=1e-6)
- Effective batch size: 4 × 8 grad accum = 32
- Mixed precision: float16 autocast + GradScaler
- Gradient clipping: max_norm=1.0

---

## Results (BCIC-IV-2a, 4-class Motor Imagery)

| Metric | Value |
|--------|-------|
| Test accuracy (keyword matching) | 31.34% |
| Chance level | 25.00% |
| Best val epoch | Epoch 6 (36.81%) |

The model generates free-form neuroscience descriptions; classification is done by counting keyword matches in the generated text.

---

## Datasets

### BCIC-IV-2a (Motor Imagery)
- 9 subjects, 4 classes: left hand / right hand / feet / tongue
- 22 EEG channels, 250 Hz → preprocessed to 200 Hz
- Window: 2–6 s post-cue → 800 samples → reshaped to (22, 4, 200)
- Splits: train (A01–A05), val (A06–A07), test (A08–A09)

### FACED (Emotion Recognition, optional)
- 9 emotion classes: Amusement, Inspiration, Joy, Tenderness, Anger, Disgust, Fear, Sadness, Neutral
- 30 EEG channels, 250 Hz
- Requires separate FACED dataset download and LMDB preparation

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch>=2.0` | Core deep learning |
| `transformers>=4.36` | TinyLlama model + tokenizer |
| `peft>=0.7` | LoRA fine-tuning |
| `bitsandbytes>=0.41` | 4-bit NF4 quantization |
| `lmdb` | Fast dataset storage |
| `einops` | Tensor reshaping |
| `scipy` | Signal filtering & resampling |

---

## Citation

If you use the CSBrain encoder, please cite:
```
CSBrain: Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding
NeurIPS 2025 Spotlight
```
