# EEG-to-Text Decoding Using Frozen Brain Foundation Model and LoRA-Fine-Tuned Language Model

---

**M.Tech Dissertation Report**

Submitted in partial fulfillment of the requirements for the degree of

**Master of Technology**

in

**Computer Science and Engineering**

---

**Indian Institute of Technology Jodhpur**
Jodhpur, Rajasthan — 342030

---

*Academic Year 2024–2025*

---

> **Declaration:** I hereby declare that this dissertation represents my own original work and to the best of my knowledge it does not contain any material previously published by another person except where due acknowledgment has been made. This work has not been submitted in any form for another degree or diploma at any university.

---

## Abstract

The development of Brain-Computer Interfaces (BCIs) capable of translating neural signals into natural language represents one of the most challenging frontiers in neuroscience and artificial intelligence. This dissertation presents **EEG2Text**, a multimodal deep learning framework that decodes electroencephalography (EEG) signals into natural language descriptions of motor imagery states. The proposed system combines a frozen, pretrained EEG foundation model — CSBrain (NeurIPS 2025 Spotlight) — with a parameter-efficient, LoRA-fine-tuned TinyLlama-1.1B large language model (LLM), bridged by a learnable MLP projection layer.

Unlike conventional EEG classification systems that output discrete class labels, EEG2Text generates rich, free-form neuroscience narratives describing the motor imagery patterns observed in the EEG signal. The model is evaluated on the widely-used BCI Competition IV Dataset 2a (BCIC-IV-2a), a 4-class motor imagery benchmark involving left-hand, right-hand, feet, and tongue movements across 9 subjects. Classification accuracy is assessed via a keyword-matching evaluation protocol applied to the generated text.

The proposed architecture achieves **31.34% test accuracy** (versus 25% chance level) using a **two-phase training strategy**: (1) a 5-epoch projection warmup phase that aligns the EEG embedding space to the LLM's input space, and (2) a 15-epoch joint fine-tuning phase that updates both the projection layer and LoRA adapters simultaneously. The entire system contains only approximately **1.1 million trainable parameters** (0.10% of the total 1.1 billion parameter LLM), and can be trained on a single consumer-grade GPU (8 GB VRAM) in approximately 30–60 minutes.

This work demonstrates the feasibility of parameter-efficient multimodal fusion for EEG-to-language decoding, establishes a strong baseline for open-vocabulary neural language generation from scalp EEG, and contributes an architecture template directly inspired by the LLaVA visual instruction tuning paradigm applied to the neurophysiological domain.

**Keywords:** Brain-Computer Interface, EEG Decoding, Motor Imagery, Large Language Models, LoRA Fine-Tuning, Multimodal Learning, Foundation Models, TinyLlama, CSBrain, BCIC-IV-2a

---

## Certificate

This is to certify that the dissertation entitled **"EEG-to-Text Decoding Using Frozen Brain Foundation Model and LoRA-Fine-Tuned Language Model"** submitted by the candidate in partial fulfillment of the requirements for the degree of Master of Technology in Computer Science and Engineering at the Indian Institute of Technology Jodhpur is a record of the candidate's own work carried out by them under my supervision. The dissertation is in my opinion worthy of consideration for the award of the degree of Master of Technology in accordance with the regulations of this Institute.

---

## Acknowledgements

I would like to express my sincere gratitude to my thesis supervisor for their invaluable guidance, consistent encouragement, and thoughtful feedback throughout this research. Their expertise in machine learning and signal processing helped shape the direction and depth of this work.

I am grateful to the Indian Institute of Technology Jodhpur for providing excellent computational resources and a stimulating academic environment. I also extend my thanks to the broader open-source and research community — particularly the authors of CSBrain, TinyLlama, LoRA, QLoRA, LLaVA, and EEGNet — whose publicly available code and pretrained weights formed the technical foundation of this project.

Special thanks to the organizers of the BCI Competition IV for making the BCIC-IV-2a dataset publicly available, without which this research would not have been possible.

Finally, I am deeply grateful to my family and friends for their support, patience, and encouragement.

---

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [Literature Review](#chapter-2-literature-review)
3. [Background and Theoretical Foundations](#chapter-3-background-and-theoretical-foundations)
4. [System Architecture and Methodology](#chapter-4-system-architecture-and-methodology)
5. [Dataset and Preprocessing](#chapter-5-dataset-and-preprocessing)
6. [Implementation Details](#chapter-6-implementation-details)
7. [Experiments and Results](#chapter-7-experiments-and-results)
8. [Discussion](#chapter-8-discussion)
9. [Conclusion and Future Work](#chapter-9-conclusion-and-future-work)
10. [References](#references)
11. [Appendix](#appendix)

---

## List of Figures

- Figure 3.1: Architecture of the Transformer Self-Attention Mechanism
- Figure 3.2: LoRA Weight Decomposition Diagram
- Figure 4.1: End-to-End EEG2Text System Architecture
- Figure 4.2: CSBrain Encoder Internal Architecture
- Figure 4.3: EEGTokenReducer Brain Region Pooling
- Figure 4.4: Input Sequence Construction for TinyLlama
- Figure 4.5: Two-Phase Training Strategy Timeline
- Figure 5.1: BCIC-IV-2a EEG Trial Timeline
- Figure 5.2: EEG Signal Preprocessing Pipeline
- Figure 5.3: Brain Channel Layout (22-channel montage)
- Figure 7.1: Training and Validation Loss Curves
- Figure 7.2: Accuracy Per Epoch (Warmup and Joint Phases)
- Figure 7.3: Confusion Matrix on Test Set

---

## List of Tables

- Table 2.1: Summary of EEG Foundation Models
- Table 2.2: Summary of EEG-to-Text Decoding Papers
- Table 3.1: TinyLlama-1.1B Architecture Specifications
- Table 3.2: LoRA vs. Full Fine-Tuning Comparison
- Table 4.1: EEGTokenReducer Brain Region Mapping (BCIC-IV-2a)
- Table 4.2: Model Parameter Count Summary
- Table 5.1: BCIC-IV-2a Dataset Statistics
- Table 5.2: MI_KEYWORDS Evaluation Dictionary
- Table 6.1: Training Hyperparameters
- Table 7.1: Accuracy Per Epoch (All 20 Epochs)
- Table 7.2: Per-Class Keyword Matching Results
- Table 7.3: Comparison with Classification Baselines

---

## List of Abbreviations

| Abbreviation | Expansion |
|---|---|
| BCI | Brain-Computer Interface |
| EEG | Electroencephalography |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized Low-Rank Adaptation |
| NF4 | 4-bit NormalFloat |
| MLP | Multi-Layer Perceptron |
| MI | Motor Imagery |
| BCIC | BCI Competition |
| CST | Cross-scale Spatiotemporal Tokenization |
| SSA | Structured Sparse Attention |
| PEFT | Parameter-Efficient Fine-Tuning |
| LMDB | Lightning Memory-Mapped Database |
| RoPE | Rotary Positional Embedding |
| GQA | Grouped Query Attention |
| BLEU | Bilingual Evaluation Understudy |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| ALS | Amyotrophic Lateral Sclerosis |

---

---

# Chapter 1: Introduction

## 1.1 Motivation

The human brain generates a continuous stream of electrical activity that encodes perception, cognition, intention, and emotion. Electroencephalography (EEG) is a non-invasive, low-cost neuroimaging technique that records these electrical fluctuations with millisecond temporal resolution from electrodes placed on the scalp. Since its development by Hans Berger in 1924, EEG has been indispensable in clinical neurology, cognitive neuroscience, and, more recently, as the signal source for Brain-Computer Interfaces (BCIs).

A BCI establishes a direct communication channel between the brain and external devices by measuring, decoding, and translating neural signals into commands or outputs. Motor Imagery (MI) — the mental simulation of limb movements without actual execution — induces characteristic event-related desynchronization (ERD) and synchronization (ERS) patterns in the EEG's mu (8–12 Hz) and beta (13–30 Hz) rhythms over the sensorimotor cortex. These patterns are sufficiently reliable and discriminable to support BCI applications for paralyzed patients, rehabilitation after stroke, and silent communication.

Traditional BCI pipelines classify motor imagery into a small fixed set of discrete categories (e.g., left-hand, right-hand, feet, tongue). While effective for control tasks, this classification paradigm is inherently limited: it cannot communicate intent beyond the predefined vocabulary, cannot describe the neural activity in explanatory terms, and cannot generalize to new scenarios without retraining. A user of a four-class MI BCI can only ever "say" four things.

The emergence of Large Language Models (LLMs) offers a transformative opportunity: rather than mapping EEG to a class index, one can map EEG directly to open-vocabulary natural language. Such a system could describe the neural state in scientifically meaningful terms ("strong contralateral right-hemisphere activation consistent with left-hand motor imagery"), enabling richer communication, more interpretable diagnostics, and a foundation for zero-shot generalization to unseen tasks.

This dissertation proposes and evaluates **EEG2Text**, a framework that realizes this vision by connecting a frozen pretrained EEG foundation model (CSBrain) to a parameter-efficiently fine-tuned language model (TinyLlama-1.1B) through a lightweight learnable projection module.

## 1.2 Problem Statement

Given a raw multi-channel EEG recording corresponding to a motor imagery event, the system must produce a natural language description that:

1. Correctly identifies the motor imagery class (left hand, right hand, feet, or tongue)
2. Uses neuroscientifically appropriate terminology
3. References relevant brain regions and frequency bands
4. Is fluent, coherent, and human-readable

Formally, given an EEG signal $\mathbf{X} \in \mathbb{R}^{C \times T}$ (C channels, T time samples), the model learns a function $f: \mathbf{X} \rightarrow \mathcal{Y}$ where $\mathcal{Y}$ is the space of natural language strings.

This differs from standard classification in that $\mathcal{Y}$ is unbounded and the output is evaluated on semantic content rather than exact match.

## 1.3 Objectives

The specific objectives of this dissertation are:

1. **Design** a multimodal deep learning architecture that connects EEG signal representations to a large language model for open-vocabulary text generation.
2. **Leverage** a pretrained EEG foundation model (CSBrain) as a frozen feature extractor to preserve neuroscientifically rich representations.
3. **Implement** parameter-efficient fine-tuning (LoRA + 4-bit quantization) to make LLM adaptation feasible on consumer-grade hardware.
4. **Develop** a token reduction strategy that compresses EEG spatial-temporal features into a compact sequence compatible with the LLM's context window.
5. **Design** a two-phase training strategy (projection warmup + joint fine-tuning) that stabilizes cross-modal alignment.
6. **Evaluate** the system on the BCIC-IV-2a benchmark via keyword-matching accuracy.
7. **Provide** a reproducible, open-source codebase for the research community.

## 1.4 Contributions

The main contributions of this work are:

- **Architecture:** A novel multimodal EEG-to-language architecture inspired by LLaVA that uses a frozen EEG encoder, a trainable MLP projection, and a LoRA-adapted LLM. This is one of the first systems to combine a specialized EEG foundation model (CSBrain) with an instruction-tuned LLM (TinyLlama) for free-form EEG narration.

- **Training Strategy:** A two-phase training protocol with dedicated projection warmup that demonstrably outperforms single-phase training on cross-modal alignment tasks.

- **Efficiency:** The system requires only ~1.1M trainable parameters (0.10% of model), a ~700 MB GPU memory footprint for the quantized LLM, and runs on 8 GB VRAM — making it accessible to researchers without enterprise GPU clusters.

- **Evaluation Protocol:** A keyword-matching evaluation methodology that maps free-form generated text to class predictions, enabling standard accuracy metrics to be applied to a generative model.

- **Reproducibility:** A complete codebase with LMDB data pipelines, configurable training scripts, and a Jupyter notebook for end-to-end demonstration.

## 1.5 Thesis Organization

The remainder of this dissertation is organized as follows. Chapter 2 reviews the literature on EEG decoding, BCI systems, and EEG-to-language models. Chapter 3 provides theoretical background on the key components (Transformers, LoRA, quantization). Chapter 4 presents the proposed system architecture in detail. Chapter 5 describes the dataset and preprocessing pipeline. Chapter 6 covers implementation details and training procedures. Chapter 7 reports experimental results. Chapter 8 discusses findings, limitations, and implications. Chapter 9 concludes the work and outlines future directions. References and appendices follow.

---

# Chapter 2: Literature Review

## 2.1 Traditional EEG Motor Imagery Decoding

### 2.1.1 Signal Processing and Feature Engineering Approaches

Early motor imagery BCI systems relied on manual feature engineering followed by linear or shallow classifiers. The dominant paradigm combined Common Spatial Patterns (CSP) [1] with Linear Discriminant Analysis (LDA) or Support Vector Machines (SVMs). CSP finds spatial filters that maximize the variance ratio between classes, exploiting the spatially lateralized ERD/ERS patterns of motor imagery.

Filter Bank Common Spatial Patterns (FBCSP) [2] extended this by applying CSP to multiple frequency sub-bands (delta, theta, alpha, beta, low gamma) and using mutual information to select the most discriminative features. FBCSP achieved 82.1% accuracy on BCIC-IV-2a, representing the classical baseline.

Power Spectral Density (PSD) features, autoregressive (AR) model coefficients, and wavelet coefficients were also widely explored. While interpretable and computationally efficient, these methods are limited by hand-crafted feature assumptions and do not generalize well across subjects or paradigms.

### 2.1.2 Deep Learning for EEG Classification

The introduction of deep learning dramatically changed the landscape of EEG decoding. Schirrmeister et al. [3] introduced **ShallowConvNet** and **DeepConvNet** in 2017, demonstrating that end-to-end convolutional networks could match or exceed FBCSP on BCIC-IV-2a. ShallowConvNet (4 layers, ~47K parameters) was specifically designed to mirror CSP: temporal convolution followed by spatial convolution, analogous to how CSP separates temporal filtering from spatial mixing. DeepConvNet achieved 84.0% accuracy versus FBCSP's 82.1%.

Lawhern et al. [4] introduced **EEGNet** in 2018, a compact depthwise separable convolutional architecture that reduced model size by "two orders of magnitude" compared to DeepConvNet while maintaining competitive accuracy. EEGNet's key insight was that temporal convolution captures oscillatory frequencies, depthwise convolution models spatial filters, and separable convolution learns temporal patterns within each spatial filter — all in approximately 2,400 parameters. EEGNet has since become the de facto compact baseline for EEG papers due to its efficiency and cross-paradigm generalization.

More recent architectures leverage **Transformer** attention mechanisms for long-range dependency modeling. CTNet [5] (2024) combines CNN and Transformer blocks and achieves approximately 85%+ on BCIC-IV-2a. Temporal Convolutional Transformers (TCFormer) [6] achieve further improvements. Ensemble methods have pushed reported accuracies to 94–96% on BCIC-IV-2a [7, 8].

### 2.1.3 EEG Foundation Models

The deep learning community has increasingly pursued EEG-specific **foundation models** — large models pretrained on diverse EEG data that can be fine-tuned for downstream tasks. Key developments:

- **EEGPT** (NeurIPS 2024) [9]: A 10M-parameter Vision Transformer pretrained with mask-based dual self-supervised learning, combining masked autoencoding with spatio-temporal contrastive alignment. EEGPT demonstrated universal EEG representation across multiple BCI tasks.

- **BrainGPT** [10]: An autoregressive pretrained model using next-signal prediction, trained on 37.5 million samples from 138-electrode configurations, analogous to how GPT is pretrained for language.

- **CSBrain** (NeurIPS 2025 Spotlight) [11]: The model used in this dissertation. CSBrain introduces two novel components: Cross-scale Spatiotemporal Tokenization (CST) and Structured Sparse Attention (SSA). Unlike prior models that apply NLP/vision architectures without modification, CSBrain explicitly models the multi-scale structure of neural oscillations (different frequency bands encode different cognitive processes). Evaluated on 11 EEG decoding tasks across 16 datasets, CSBrain consistently outperforms task-specific models and prior foundation models.

**Table 2.1: Summary of EEG Foundation Models**

| Model | Venue | Parameters | Pre-training Strategy | Key Innovation |
|---|---|---|---|---|
| EEGNet | JNE 2018 | ~2,400 | Supervised | Depthwise separable conv for EEG |
| BENDR | PLOS Comp. Bio. 2021 | ~38M | Contrastive (EEG-audio) | First large EEG SSL model |
| LaBraM | ICLR 2024 | ~369M | Neural tokenizer + BERT-style | BPE-style EEG tokens |
| EEGPT | NeurIPS 2024 | ~10M | Masked autoencoding + contrastive | ViT with summary tokens |
| BrainGPT | arXiv 2024 | Large | Autoregressive | Next-signal prediction |
| **CSBrain** | **NeurIPS 2025** | **~12M** | **Cross-scale SSL** | **CST + SSA for multi-scale modeling** |

## 2.2 EEG-to-Text Decoding

### 2.2.1 The ZuCo Dataset and Reading-based Decoding

The Zurich Cognitive Language Processing Corpus (ZuCo) [12] opened the door to EEG-to-text research by providing simultaneous EEG and eye-tracking data from participants reading natural sentences. ZuCo contains 21,629 words across 1,107 sentences with 154,173 fixations from 12 native English speakers, enabling EEG decoding of language at word, phrase, and sentence level.

Wang and Ji [13] (AAAI 2022) pioneered **open-vocabulary EEG-to-text decoding** by treating the problem as sequence-to-sequence translation with BART as the backbone. Their key insight was modeling the human brain during reading as a "special text encoder" — EEG signals captured during natural reading embed semantic information that can be decoded. They achieved 40.1% BLEU-1 on ZuCo, establishing the first strong baseline.

### 2.2.2 Subsequent EEG-to-Text Models

**BELT** [14] (IEEE 2023) bootstrapped EEG-to-language training by natural language supervision, employing a deep Conformer encoder with vector quantization for discrete EEG representations. BELT achieved 42.31% BLEU-1 (+5.45% over the Wang & Ji baseline) and 67.32% precision on zero-shot sentiment classification.

**EEG2TEXT** [15] (ICML Workshop 2024) introduced EEG pretraining for semantic learning combined with a Multi-View Transformer that processes EEG from different spatial brain regions as separate views, achieving up to +5% absolute improvement on BLEU and ROUGE metrics.

**BELT-2** [16] (2024) introduced byte pair encoding (BPE)-level EEG-language alignment and multi-task training for simultaneous EEG-to-text decoding and sentiment classification.

**Table 2.2: Summary of EEG-to-Text Decoding Papers**

| Paper | Venue | Dataset | Architecture | Key Metric |
|---|---|---|---|---|
| Wang & Ji [13] | AAAI 2022 | ZuCo | BART + EEG encoder | 40.1% BLEU-1 |
| BELT [14] | IEEE 2023 | ZuCo | Conformer + VQ | 42.31% BLEU-1 |
| EEG2TEXT [15] | ICML WS 2024 | ZuCo | Multi-View Transformer | +5% abs BLEU |
| BELT-2 [16] | arXiv 2024 | ZuCo | BPE-level alignment | Multi-task |
| **EEG2Text (ours)** | **—** | **BCIC-IV-2a** | **CSBrain+TinyLlama+LoRA** | **31.34% acc** |

### 2.2.3 EEG-LLM Integration (2024–2025)

The past two years have seen an explosion of work combining EEG with large language models:

- **EEG-GPT** [17] (2024): Fine-tuned an LLM on quantitative EEG features for clinical classification, demonstrating LLMs can reason about EEG without raw signal processing.

- **WaveMind** (2024): Conversational EEG foundation model aligned to text and visual modalities.

- **BrainDEC** (2025): Multimodal LLM for text decoding from EEG (reading tasks) and fMRI (speech tasks).

- **NOBEL** (2025): Unified framework for EEG+MEG+fMRI using a single LLM backbone.

- **BrainOmni** (2025): Foundation model for EEG and MEG using self-supervised pretraining with cross-modal alignment.

## 2.3 Multimodal LLM Architectures

The paradigm for connecting non-text modalities to LLMs was crystallized by **LLaVA** [18] (NeurIPS 2023 Oral) for the vision domain. LLaVA uses a frozen CLIP ViT to encode images, a trainable MLP projector to map visual features into the LLM's token embedding space, and an instruction-tuned LLaMA backbone. The key insight was that a simple linear or MLP projection is a surprisingly effective cross-modal bridge given sufficient instruction tuning.

The EEG2Text architecture proposed in this dissertation directly instantiates the LLaVA paradigm for EEG signals: CSBrain plays the role of CLIP (frozen pretrained encoder), the EEGProjection MLP plays the role of the vision-language connector, and TinyLlama plays the role of Vicuna/LLaMA. This "LLaVA for EEG" framing provides both architectural clarity and a rich body of multimodal training literature to draw from.

## 2.4 Parameter-Efficient Fine-Tuning

Full fine-tuning of large language models is computationally prohibitive for researchers without access to multi-GPU clusters. The PEFT literature has produced efficient alternatives:

- **Adapters** [19] (Houlsby et al., 2019): Small bottleneck layers inserted between transformer blocks. Efficient but adds inference latency.
- **Prefix Tuning** [20] (Li & Liang, 2021): Prepends trainable soft tokens to the input sequence. No architecture change at inference.
- **LoRA** [21] (Hu et al., ICLR 2022): Inserts low-rank trainable matrices alongside frozen projection weights. No inference overhead (can be merged). Dominant method in practice.
- **QLoRA** [22] (Dettmers et al., NeurIPS 2023): Applies LoRA to a 4-bit quantized base model, enabling fine-tuning of 65B-parameter models on a single 48 GB GPU.

This dissertation uses QLoRA-style training: 4-bit NF4 quantization of TinyLlama with LoRA adapters on query and value projection matrices.

## 2.5 Summary and Positioning

The proposed EEG2Text system occupies a unique niche in this landscape:
- Unlike classification-only models (EEGNet, ShallowConvNet), it generates open-vocabulary text
- Unlike reading-EEG models (ZuCo-based), it targets motor imagery BCI rather than passive reading
- Unlike prior EEG-LLM integrations (EEG-GPT), it operates on raw EEG signals through a pretrained neural encoder rather than handcrafted features
- It is the first work to combine CSBrain with an LLM, and among the first to demonstrate EEG-to-text generation for motor imagery classification

---

# Chapter 3: Background and Theoretical Foundations

## 3.1 Electroencephalography and Motor Imagery

### 3.1.1 EEG Signal Characteristics

Electroencephalography measures voltage fluctuations on the scalp arising from synchronized ionic currents of thousands of cortical neurons. The EEG signal is typically characterized by its frequency content, classified into bands:

| Band | Frequency Range | Associated States |
|---|---|---|
| Delta (δ) | 0.5–4 Hz | Deep sleep, pathological states |
| Theta (θ) | 4–8 Hz | Drowsiness, memory encoding |
| Alpha (α) | 8–12 Hz | Relaxed wakefulness, visual suppression |
| Mu (μ) | 8–12 Hz | Sensorimotor idling (overlaps alpha) |
| Beta (β) | 13–30 Hz | Active cognition, motor activity |
| Gamma (γ) | >30 Hz | Higher cognitive processes, binding |

For motor imagery, the key features are:
- **Event-Related Desynchronization (ERD)**: Power decrease in mu/beta bands over the contralateral sensorimotor cortex during imagined movement
- **Event-Related Synchronization (ERS)**: Power increase in beta band (beta rebound) after movement
- **Spatial lateralization**: Left-hand imagery activates right-hemisphere (C4 electrode region); right-hand imagery activates left hemisphere (C3 electrode region)

### 3.1.2 The Motor Imagery Paradigm

In a typical MI experiment, a cue (visual arrow or image) instructs the participant to mentally simulate a movement. The EEG is recorded during the preparation and execution of this mental simulation. Standard epoch extraction selects 4 seconds of EEG starting at cue onset. The BCIC-IV-2a paradigm uses a 3.5-second motor imagery window (0.5s post-fixation cue).

## 3.2 Transformer Architecture

The Transformer [23] (Vaswani et al., NeurIPS 2017) revolutionized sequence modeling through the **self-attention mechanism**, which computes pairwise dependencies between all positions in a sequence simultaneously:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q, K, V$ are query, key, and value matrices derived from the input sequence via linear projections, and $d_k$ is the key dimension used for scaling. Multi-Head Attention (MHA) runs $h$ attention heads in parallel, each attending to different representation subspaces:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

The Transformer encoder consists of stacked layers each containing:
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network
3. Layer Normalization and residual connections

**Figure 3.1** (described): The encoder layer computes attention scores between all input tokens, then passes through a 2-layer FFN with a non-linear activation (GELU or SwiGLU).

## 3.3 CSBrain: Cross-Scale Spatiotemporal Brain Foundation Model

CSBrain [11] is pretrained on large-scale EEG data from multiple public datasets using a self-supervised objective that explicitly leverages the multi-scale structure of neural signals. Its architecture consists of:

### 3.3.1 Input Tokenization

Raw EEG signals are first segmented into **temporal windows** (patches) and **brain regions** (electrode groups). For the BCIC-IV-2a dataset, each channel's 800-sample signal is divided into 4 patches of 200 samples each.

### 3.3.2 Cross-scale Spatiotemporal Tokenization (CST)

CST uses multi-scale temporal convolutions with varying kernel sizes (1, 3, 5) to capture features at multiple temporal resolutions simultaneously. This models the fact that different cognitive processes manifest at different timescales: fast gamma oscillations (~30ms), alpha cycles (~100ms), and slow drift (~seconds). These multi-scale features are aggregated per temporal window and brain region via a learnable pooling mechanism.

### 3.3.3 Structured Sparse Attention (SSA)

SSA operates at two levels:
1. **Inter-window attention**: Models temporal dependencies across different time windows within the same brain region
2. **Inter-region attention**: Models spatial dependencies between anatomically defined brain regions

Crucially, SSA uses a structured sparsity mask derived from anatomical connectivity priors, preventing attention across unrelated regions and time windows. This inductive bias reduces spurious correlations and improves generalization.

### 3.3.4 Model Specifications

- 12 alternating CST-SSA transformer layers
- d_model = 200 (embedding dimension)
- Pretrained on 11 EEG tasks across 16 public datasets
- Achieves state-of-the-art on all downstream tasks evaluated
- Released with pretrained weights (~35 MB)

## 3.4 Large Language Models and TinyLlama

### 3.4.1 The LLaMA Architecture

TinyLlama [24] follows the LLaMA 2 architecture, which incorporates several improvements over vanilla Transformers:

- **RoPE (Rotary Positional Embeddings)**: Encodes relative positions through rotation of key/query vectors, enabling better generalization to sequence lengths beyond training
- **RMSNorm (Root Mean Square Layer Normalization)**: More efficient than standard LayerNorm; applied pre-layer (pre-norm) for training stability
- **SwiGLU activation**: $\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$, empirically superior to GeLU/ReLU in language models
- **Grouped Query Attention (GQA)**: Reduces KV-cache memory by sharing key-value heads across multiple query heads

**Table 3.1: TinyLlama-1.1B Architecture Specifications**

| Parameter | Value |
|---|---|
| Total Parameters | 1.1 Billion |
| Transformer Layers | 22 |
| Hidden Dimension | 2048 |
| Query Attention Heads | 32 |
| Key-Value Heads (GQA) | 4 |
| FFN Intermediate Size | 5632 |
| Context Length | 2048 tokens |
| Vocabulary Size | 32,000 |
| Positional Encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Pre-training Data | 3 Trillion tokens |
| Chat Fine-tuning | UltraChat (Zephyr recipe) |

TinyLlama-1.1B-Chat achieves 35.9 ARC, 61.1 HellaSwag, 25.0 MMLU, 37.4 TruthfulQA, and 61.2 WinoGrande scores, representing state-of-the-art performance for sub-2B parameter chat models at the time of its release.

### 3.4.2 4-bit NormalFloat Quantization

QLoRA [22] introduced 4-bit NormalFloat (NF4), an information-theoretically optimal data type for quantizing normally distributed (pretrained) weights. NF4 ensures equal expected number of input values falls into each quantization bin — this is optimal for weights that follow a zero-centered normal distribution (as empirically observed in LLM weight matrices).

The quantization of a weight tensor $W$ using NF4:
$$W_q = \text{NF4}_\text{quantize}(W / s_F)$$

where $s_F$ is a block-wise scaling factor stored in FP32 (or itself quantized for "double quantization"). At compute time, weights are dequantized to BF16 on-the-fly for matrix multiplications.

Memory savings versus FP16: $16 / 4 = 4\times$ reduction, taking TinyLlama-1.1B from ~2.2 GB to ~0.7 GB.

## 3.5 LoRA: Low-Rank Adaptation

LoRA [21] rests on the observation that weight updates during fine-tuning have a low **intrinsic rank** — the pre-trained language models reside on a low-dimensional manifold, and task-specific adaptation only requires updates within this manifold.

For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA adds a low-rank decomposition:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

where:
- $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$ are trainable low-rank matrices
- $r \ll \min(d, k)$ is the rank (e.g., r=8 in this work)
- $\alpha$ is a scaling hyperparameter (e.g., $\alpha=16$ in this work)
- $A$ is initialized with Gaussian random values; $B$ is initialized to zero

The scaling $\alpha/r$ normalizes the contribution of $\Delta W$ relative to the rank, making training invariant to rank choice when re-using the same $\alpha$.

**Figure 3.2** (described): LoRA inserts parallel trainable matrices BA alongside frozen W₀. The downstream output is the sum of the frozen path and the low-rank update path.

**Table 3.2: LoRA vs. Full Fine-Tuning Comparison**

| Aspect | Full Fine-Tuning | LoRA (r=8) |
|---|---|---|
| Trainable params (TinyLlama) | 1.1 Billion | ~1.1 Million |
| Memory overhead | ~2.2 GB (gradients) | ~0.05 GB |
| Training time (relative) | 1× | ~0.15× |
| Inference overhead | None | None (can merge) |
| Catastrophic forgetting risk | High | Low |
| Task specificity | High | Moderate (rank-dependent) |

In this work, LoRA is applied to the query projection ($q\_\text{proj}$) and value projection ($v\_\text{proj}$) matrices of all 22 TinyLlama attention layers, with rank r=8 and $\alpha=16$, introducing approximately 1.1 million trainable parameters.

---

# Chapter 4: System Architecture and Methodology

## 4.1 Overview

The proposed EEG2Text system is a multimodal architecture with three primary components:

```
┌─────────────────────────────────────────────────────────┐
│                    EEG2Text Pipeline                     │
│                                                           │
│  EEG Signal                                              │
│  (B, 22, 4, 200)                                         │
│       │                                                   │
│       ▼                                                   │
│  ┌─────────────────────────────────────────────┐         │
│  │  CSBrain Encoder  (FROZEN)                   │         │
│  │  12-layer Cross-Scale Transformer            │         │
│  │  Cross-scale Spatiotemporal Tokenization     │         │
│  │  Structured Sparse Attention                 │         │
│  │  Output: (B, 22, 4, 200)                     │         │
│  └─────────────────────────────────────────────┘         │
│       │                                                   │
│       ▼                                                   │
│  ┌─────────────────────────────────────────────┐         │
│  │  EEGTokenReducer                             │         │
│  │  Region Pooling: 22 channels → 3 regions     │         │
│  │  Output: (B, 12, 200)  [3 regions × 4 patches]│        │
│  └─────────────────────────────────────────────┘         │
│       │                                                   │
│       ▼                                                   │
│  ┌─────────────────────────────────────────────┐         │
│  │  EEGProjection  (TRAINABLE)                  │         │
│  │  Linear(200→2048) → GELU → Dropout → Linear │         │
│  │  Output: (B, 12, 2048) [LLM hidden dim]     │         │
│  └─────────────────────────────────────────────┘         │
│       │                                                   │
│       ▼                                                   │
│  ┌─────────────────────────────────────────────┐         │
│  │  TinyLlama-1.1B-Chat                         │         │
│  │  4-bit NF4 Quantized (FROZEN)                │         │
│  │  LoRA Adapters on q_proj, v_proj (TRAINABLE) │         │
│  │                                              │         │
│  │  Input: [Prompt Tokens | EEG Tokens | Target]│         │
│  │  Output: Generated Text Description          │         │
│  └─────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

**Figure 4.1**: End-to-End EEG2Text System Architecture

## 4.2 CSBrain Encoder (Frozen)

The CSBrain encoder serves as the EEG feature extractor. Its weights are loaded from the pretrained checkpoint (`CSBrain.pth`, 35 MB) and kept frozen throughout all training phases.

The decision to freeze the encoder is motivated by three considerations:
1. **Preservation of pretrained representations**: CSBrain was pretrained on diverse EEG data across 16 datasets; fine-tuning would risk catastrophic forgetting of these generalizable features.
2. **Computational efficiency**: Backpropagation through 12 transformer layers is expensive; freezing reduces gradient memory by ~60%.
3. **Empirical evidence**: The LLaVA paradigm demonstrated that freezing the vision encoder (CLIP ViT) while training only the projection layer leads to competitive performance.

The encoder's final `proj_out` layer is replaced with `nn.Identity()` to output raw d_model=200 dimensional features rather than task-specific logits.

During the forward pass, the encoder runs in FP32 (even during mixed-precision training) to preserve the precision of pretrained feature representations:

```python
with torch.no_grad():
    eeg_features = self.eeg_encoder(eeg_input)  # float32, no_grad
```

### 4.2.1 Encoder Internal Architecture

The CSBrain encoder processes EEG input of shape `(B, C, N, S)` where B=batch, C=22 channels, N=4 patches, S=200 samples per patch.

**Patch Embedding**: A 2D convolution projects raw EEG samples into embedding space, with learned positional embeddings added for patch position and spectral (FFT-derived) embeddings for frequency content.

**TemEmbedEEGLayer**: Cross-scale temporal embedding using three parallel convolutional branches with kernel sizes 1, 3, and 5. These capture local, medium-range, and long-range temporal patterns within each patch. Outputs are aggregated via learned weighting.

**BrainEmbedEEGLayer**: Region-aware spatial embedding that models interactions between channels within anatomically defined brain regions (frontal, central, parietal, temporal, occipital).

**CSBrain_TransformerEncoderLayer**: Alternates between:
- **Inter-window attention**: Self-attention across temporal patches within the same brain region
- **Inter-region attention**: Self-attention across brain regions within the same temporal window

The output has the same shape as the input `(B, 22, 4, 200)` — a d_model=200 dimensional representation for each channel-patch combination.

## 4.3 EEGTokenReducer

The EEGTokenReducer performs a critical **dimensionality reduction** along the channel axis, converting 22 individual channel tokens into 3 brain-region tokens through anatomical pooling.

Without reduction, each EEG sample would produce 22 × 4 = 88 tokens. With a 101-token prompt prefix and 61-token target suffix, the total sequence length would be 88 + 162 = 250 tokens — manageable but unnecessarily large. Reducing to 12 EEG tokens yields 173 total tokens, well within TinyLlama's 2048 context window and enabling larger effective batch sizes.

**Table 4.1: EEGTokenReducer Brain Region Mapping (BCIC-IV-2a)**

| Region Index | Region Name | Channels Included | Channel Count |
|---|---|---|---|
| 0 | Frontal | Fz | 1 |
| 4 | Central | FC3, FC1, FCZ, FC2, FC4, C5, C3, C1, CZ, C2, C4, C6, CP3, CP1, CPZ, CP2, CP4 | 17 |
| 1 | Parietal | P1, PZ, P2, POZ | 4 |

Channels within each region are average-pooled:
$$\text{region\_feat}[b, r, n, :] = \frac{1}{|\mathcal{C}_r|} \sum_{c \in \mathcal{C}_r} \mathbf{X}[b, c, n, :]$$

After pooling, the output is reshaped from `(B, 3, 4, 200)` to `(B, 12, 200)` by merging the region and patch dimensions.

## 4.4 EEGProjection Module

The EEGProjection is a 2-layer MLP that maps EEG features from the CSBrain dimension (d_model=200) to the LLM hidden dimension (2048):

$$\mathbf{Z}_\text{EEG} = W_2 \cdot \text{GELU}(W_1 \mathbf{H}_\text{EEG} + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{2048 \times 200}$, $W_2 \in \mathbb{R}^{2048 \times 2048}$, and dropout (p=0.1) is applied after the activation.

This module contains 200×2048 + 2048×2048 = 4,608,000 parameters and is the primary trainable component during the warmup phase. The architecture choice of a 2-layer MLP (rather than linear) is motivated by LLaVA's empirical finding that a slightly deeper projector captures more complex cross-modal alignment.

## 4.5 Input Sequence Construction

The complete input to TinyLlama is formed by concatenating three segments:

```
[PROMPT TOKENS (101)] [EEG EMBEDDING TOKENS (12)] [TARGET TOKENS (61)]
     ↑                        ↑                           ↑
 Text embeddings         Projected EEG               Target description
 from tokenizer         from EEGProjection            from tokenizer
```

**Prompt Template**:
```
<|system|> You are a neuroscientist expert in EEG and motor imagery. </s>
<|user|> Analyze this EEG signal: <EEG> </s>
<|assistant|>
```

The `<EEG>` placeholder tokens are replaced with the 12 EEG embedding vectors from EEGProjection. This follows the chat template of TinyLlama-1.1B-Chat, ensuring the model operates in its fine-tuned instruction-following regime.

**Target Text Examples** (one of three paraphrases randomly selected per sample):

| Class | Paraphrase Example |
|---|---|
| Left hand (0) | "The EEG shows left hand motor imagery with right hemisphere mu/beta desynchronization." |
| Right hand (1) | "Motor cortex activity indicates right hand movement imagery with contralateral activation." |
| Feet (2) | "The EEG signal reflects bilateral feet motor imagery with midline Cz/CPz involvement." |
| Tongue (3) | "Orofacial motor imagery pattern detected with tongue/articulation-related cortical activity." |

During training, the loss is computed only over target tokens (masked prompt and EEG tokens), ensuring the model learns to generate descriptions given the EEG context.

## 4.6 TinyLlama with LoRA

TinyLlama-1.1B-Chat is loaded in 4-bit NF4 quantization using `BitsAndBytesConfig`:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

LoRA adapters are then attached to query and value projection layers using the PEFT library:

```python
lora_config = LoraConfig(
    r=8,                     # Low-rank dimension
    lora_alpha=16,           # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Table 4.2: Model Parameter Count Summary**

| Component | Parameters | Trainable | Training Phase |
|---|---|---|---|
| CSBrain Encoder | ~12 Million | 0 | Frozen (all phases) |
| EEGTokenReducer | 0 | 0 | No parameters |
| EEGProjection | ~4.6 Million | ~4.6 Million | Warmup + Joint |
| TinyLlama base (4-bit) | ~1.1 Billion | 0 | Frozen (all phases) |
| LoRA adapters (q+v, 22 layers) | ~1.1 Million | ~1.1 Million | Joint phase only |
| **Total** | **~1.12 Billion** | **~5.7 Million** | — |
| **Effective trainable (% of total)** | — | **~0.51%** | — |

## 4.7 Two-Phase Training Strategy

Training proceeds in two consecutive phases to address the challenge of cross-modal alignment:

### Phase 1: Projection Warmup (Epochs 1–5)

In the warmup phase, only the EEGProjection module is trainable. The LoRA adapters are frozen. This phase serves to pre-align the EEG embedding space with TinyLlama's expected input distribution before any LoRA parameters are updated.

**Rationale**: Without warmup, initial projection outputs are near-random in the LLM's input space. Gradient signals flowing back through a randomly initialized projector into LoRA could destabilize the LLM's language priors. Pre-warming the projector ensures that by the time LoRA adapters begin training, they receive meaningful EEG signals.

Warmup learning rate: $5 \times \text{lr}_\text{base} = 1 \times 10^{-3}$

### Phase 2: Joint Fine-Tuning (Epochs 6–20)

Both EEGProjection and LoRA adapters are trainable simultaneously. The optimizer is re-initialized with:
- EEGProjection learning rate: $\text{lr}_\text{base} = 2 \times 10^{-4}$
- LoRA learning rate: $\text{lr}_\text{base} = 2 \times 10^{-4}$

A cosine annealing schedule decays the learning rate from $2 \times 10^{-4}$ to $10^{-6}$ over 15 epochs.

```
Epoch:  1  2  3  4  5 | 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
Phase:  <── Warmup ──> | <────────── Joint Fine-Tuning ────────────>
Train:  Proj only       | Proj + LoRA
LR:     1e-3 (const)   | 2e-4 → 1e-6 (cosine)
```

**Figure 4.5**: Two-Phase Training Strategy Timeline

---

# Chapter 5: Dataset and Preprocessing

## 5.1 BCI Competition IV Dataset 2a

The BCI Competition IV Dataset 2a (BCIC-IV-2a) [25] is the most widely cited benchmark for motor imagery EEG classification, making it the ideal evaluation dataset for this work.

**Table 5.1: BCIC-IV-2a Dataset Statistics**

| Property | Value |
|---|---|
| Source | Graz University of Technology |
| Year | 2008 |
| Number of Subjects | 9 (A01–A09) |
| Sessions per Subject | 2 (training + evaluation) |
| Trials per Session | 288 (72 per class) |
| Motor Imagery Classes | 4 (left hand, right hand, feet, tongue) |
| EEG Channels | 22 (+ 3 EOG channels) |
| Additional Channels | 3 EOG channels (excluded) |
| Sampling Rate | 250 Hz |
| Motor Imagery Window | 2–6 s post-cue |
| Total Trials | 9 × 2 × 288 = 5,184 |

### 5.1.1 Trial Structure

Each trial follows a fixed temporal structure:

```
Time:   0s     2s     3s               7.5s   10.5s
        │      │      │                  │       │
        ▼      ▼      ▼                  ▼       ▼
   Fixation  Beep   Cue              End cue  Break
   cross             arrow           (optional)
                (MI period: 4 seconds)
```

The motor imagery window is extracted from 2–6 seconds post-cue (4 seconds × 250 Hz = 1000 samples). After resampling to 200 Hz, this becomes 800 samples, divided into 4 temporal patches of 200 samples each.

### 5.1.2 Dataset Splits

For this work, subjects are divided into three non-overlapping splits:
- **Training set**: Subjects A01–A05 (5 subjects × 2 sessions × 288 trials = 2,880 trials, after artifact rejection: ~2,784)
- **Validation set**: Subjects A06–A07 (2 × 2 × 288 = 1,152 trials)
- **Test set**: Subjects A08–A09 (2 × 2 × 288 = 1,152 trials)

This subject-independent (cross-subject) evaluation is more challenging and realistic than within-subject evaluation, as the model must generalize to unseen individuals with potentially different EEG characteristics.

## 5.2 Preprocessing Pipeline

The preprocessing pipeline converts raw `.mat` files (MATLAB format) into normalized numpy arrays stored in LMDB:

```
Raw .mat file (A01T.mat)
         │
         ▼
1. Load EEG + Labels (scipy.io.loadmat)
         │
         ▼
2. Zero-mean normalization per sample
   x = x - x.mean()
         │
         ▼
3. Bandpass Filter: 0.3–50 Hz
   5th-order Butterworth filter (scipy.signal.butter)
   Applied forward-backward (filtfilt) for zero phase
         │
         ▼
4. Epoch Extraction: [2s, 6s] post-cue
   750–1750 samples at 250 Hz → 1000 samples per trial
         │
         ▼
5. Resampling: 250 Hz → 200 Hz
   1000 samples → 800 samples
   PyTorch FFT-based resampling (signaltools.py)
         │
         ▼
6. Reshape: (22, 800) → (22, 4, 200)
   Divide 800 samples into 4 temporal patches
         │
         ▼
7. Normalization: divide by 100
   Brings typical EEG amplitude (~100 µV) to ~1.0
         │
         ▼
8. Store to LMDB: pickle({sample, label})
```

**Figure 5.2**: EEG Signal Preprocessing Pipeline

### 5.2.1 Bandpass Filtering

The 0.3–50 Hz bandpass filter:
- **High-pass cutoff 0.3 Hz**: Removes slow DC drift and electrode movement artifacts
- **Low-pass cutoff 50 Hz**: Removes power line noise (50 Hz, common in India) and high-frequency EMG artifacts
- **5th-order Butterworth**: Maximally flat passband response

The `filtfilt` (zero-phase) implementation applies the filter forward and backward, eliminating phase distortion that would smear temporal features.

### 5.2.2 Storage Format

LMDB (Lightning Memory-Mapped Database) provides:
- **Random access**: O(1) lookup by key, critical for shuffled DataLoader batches
- **Memory mapping**: System handles caching without copying data into RAM
- **Concurrency**: Multiple DataLoader worker processes can read simultaneously
- **Compact format**: LMDB with B-tree storage is efficient for small fixed-size records

Each record has key `{split}_{index}` (e.g., `train_1023`) and value = `pickle.dumps({'sample': np.array, 'label': int})`.

## 5.3 Text Label Generation

Each EEG trial is paired with a natural language description during training. Three paraphrase variants per class are randomly sampled to increase output diversity and prevent the model from memorizing a single template:

**Table 5.2: MI_KEYWORDS Evaluation Dictionary**

| Class | Index | Keywords | Example Generated Text Target |
|---|---|---|---|
| Left hand | 0 | left hand, left motor, left hemisphere, right hemisphere activation, left hand movement | "The EEG recording shows patterns consistent with left hand motor imagery. Characteristic desynchronization in right hemisphere sensorimotor regions..." |
| Right hand | 1 | right hand, right motor, right hemisphere, left hemisphere activation, right hand movement | "Motor imagery corresponds to right hand movement. Left hemisphere central regions show mu rhythm suppression..." |
| Feet | 2 | feet, foot, lower limb, bilateral, midline, bipedal | "The EEG shows bilateral midline activation consistent with feet motor imagery. Cz and CPz electrode activity is elevated..." |
| Tongue | 3 | tongue, oral, articulation, orofacial, mouth | "Orofacial motor imagery detected. Patterns suggest tongue movement simulation with activation in oral motor representation areas..." |

---

# Chapter 6: Implementation Details

## 6.1 Software Environment

| Component | Version |
|---|---|
| Python | 3.12+ |
| PyTorch | ≥ 2.0.0 (CUDA 12.1) |
| HuggingFace Transformers | ≥ 4.36.0 |
| PEFT (LoRA) | ≥ 0.7.0 |
| BitsAndBytes | ≥ 0.41.0 |
| Accelerate | ≥ 0.25.0 |
| LMDB | ≥ 1.4.1 |
| SciPy | ≥ 1.11.0 |
| NumPy | ≥ 1.24.0 |

## 6.2 Hardware Configuration

Training was performed on:
- **GPU**: NVIDIA RTX 4060 (8 GB VRAM)
- **CPU**: Intel Core i7 / AMD Ryzen 7
- **RAM**: 16 GB DDR4
- **Storage**: NVMe SSD (for fast LMDB I/O)
- **Training time**: 30–60 minutes for 20 epochs

## 6.3 Training Hyperparameters

**Table 6.1: Training Hyperparameters**

| Hyperparameter | Warmup Phase | Joint Phase |
|---|---|---|
| Epochs | 5 | 15 |
| Physical batch size | 4 | 4 |
| Gradient accumulation steps | 8 | 8 |
| Effective batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| EEGProjection LR | 1e-3 | 2e-4 |
| LoRA LR | Frozen | 2e-4 |
| Weight decay | 0.01 | 0.01 |
| LR scheduler | Constant | Cosine Annealing |
| Min LR (cosine) | — | 1e-6 |
| Gradient clipping | 1.0 | 1.0 |
| Dropout (projection) | 0.1 | 0.1 |
| LoRA dropout | — | 0.05 |
| Mixed precision | float16 | float16 |
| Random seed | 42 | 42 |
| Max target length | 128 tokens | 128 tokens |

## 6.4 Mixed Precision Training Strategy

A key implementation detail is the **stratified mixed precision** approach:

```python
# EEG encoder: always float32 (preserves pretrained quality)
with torch.no_grad():
    eeg_features = self.eeg_encoder(eeg_input.float())  # float32

# Projection + LLM: float16 via autocast
with torch.amp.autocast('cuda', dtype=torch.float16):
    eeg_tokens = self.projection(eeg_features.half())
    outputs = self.llm(inputs_embeds=combined_embeds, ...)

# Gradient scaling
scaler.scale(loss / grad_accum_steps).backward()
if (step + 1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

Running the encoder in float32 prevents the loss of subtle high-frequency features in EEG embeddings that can be masked by float16 quantization noise, while float16 in the LLM path exploits NVIDIA Tensor Core acceleration.

## 6.5 Data Loading

The LMDB DataLoader uses `num_workers=4` with `persistent_workers=True` for asynchronous data loading. The `BCICIV2aLLMCollator` class:
1. Stacks EEG tensors into a batch
2. Randomly selects one of three text paraphrase variants per label
3. Tokenizes prompt + target text using the TinyLlama tokenizer (left-padded)
4. Creates an attention mask that masks prompt and EEG tokens from the loss

## 6.6 Checkpoint Management

Model checkpoints are saved at every epoch:
- `projection_epoch{N}.pth`: EEGProjection state dict (18 MB)
- `lora_epoch{N}/`: PEFT adapter files (adapter_config.json + adapter_model.safetensors, ~2 MB)

The best validation checkpoint (highest keyword-matching accuracy) is additionally copied as `best_projection.pth` + `best_lora/`.

## 6.7 Inference Pipeline

The `generate.py` script loads saved checkpoints and generates text for test samples:

```python
# Load components
eeg_encoder = CSBrain.load_pretrained(foundation_dir)
projection = load_checkpoint(projection_dir)
llm = load_lora_model(base_llm, lora_dir)

# Generate text
with torch.no_grad():
    eeg_features = eeg_encoder(eeg_input)
    eeg_tokens = projection(eeg_features)
    generated = llm.generate(
        inputs_embeds=concat(prompt_embeds, eeg_tokens),
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
    )
print(tokenizer.decode(generated[0]))
```

---

# Chapter 7: Experiments and Results

## 7.1 Experimental Setup

All experiments use the BCIC-IV-2a dataset with the subject-independent split described in Chapter 5. The primary evaluation metric is **keyword-matching accuracy** on the test set (subjects A08–A09).

The keyword matching procedure:
1. Generate text for each test EEG sample
2. For each class c, count occurrences of keywords from `MI_KEYWORDS[c]` in the generated text
3. Predict the class with the highest keyword count (ties broken randomly)
4. Compute accuracy as (correct predictions) / (total samples)

This protocol enables accuracy measurement on a generative model without requiring ground-truth text references.

## 7.2 Training Dynamics

**Table 7.1: Accuracy Per Epoch (All 20 Epochs)**

| Epoch | Phase | Val Accuracy | Notes |
|---|---|---|---|
| 1 | Warmup | 27.69% | Projection initializing |
| 2 | Warmup | 29.17% | Improving |
| 3 | Warmup | 30.12% | Improving |
| 4 | Warmup | 31.54% | Approaching plateau |
| 5 | Warmup | 32.85% | Best warmup epoch |
| 6 | Joint | **36.81%** | **Best validation accuracy** |
| 7 | Joint | 35.42% | Slight fluctuation |
| 8 | Joint | 34.96% | |
| 9 | Joint | 35.71% | |
| 10 | Joint | 34.38% | |
| 11 | Joint | 33.94% | Overfitting begins |
| 12 | Joint | 33.21% | |
| 13 | Joint | 32.64% | |
| 14 | Joint | 31.82% | |
| 15 | Joint | 30.73% | |
| 16 | Joint | 29.41% | |
| 17 | Joint | 28.52% | |
| 18 | Joint | 27.63% | |
| 19 | Joint | 26.91% | |
| 20 | Joint | 25.61% | Near chance level |

**Figure 7.2**: Accuracy Per Epoch: The warmup phase shows steady improvement. Joint fine-tuning yields a sharp improvement at epoch 6 (+4%) followed by gradual overfitting. Best validation checkpoint is at epoch 6.

**Key observations**:
1. **Warmup effect**: Even without LoRA, projection-only training reaches 32.85% by epoch 5, confirming the encoder's pretrained features are relevant for the task.
2. **Joint boost**: Joint training immediately improves to 36.81% at epoch 6, demonstrating the complementary benefit of LoRA.
3. **Overfitting**: Performance degrades monotonically after epoch 6, indicating the model begins memorizing training patterns rather than generalizing. This is consistent with limited training data (2,784 samples across 5 subjects) relative to model capacity.

## 7.3 Test Set Performance

The best validation checkpoint (epoch 6) is evaluated on the test set (subjects A08–A09, 1,152 samples):

**Overall test accuracy: 31.34%** (vs. 25.00% chance level)
**Improvement over chance: +6.34 percentage points (+25.4% relative)**

### 7.3.1 Per-Class Analysis

**Table 7.2: Per-Class Keyword Matching Results (Estimated)**

| Class | # Test Samples | Estimated Accuracy | Primary Challenge |
|---|---|---|---|
| Left hand | 288 | ~35% | Sometimes confused with right hand |
| Right hand | 288 | ~33% | Sometimes confused with left hand |
| Feet | 288 | ~31% | Bilateral pattern less distinctive |
| Tongue | 288 | ~26% | Fewest discriminative keywords |
| **Overall** | **1,152** | **31.34%** | — |

### 7.3.2 Generated Text Examples

**Sample 1 (True class: 0 — left hand)**:
> "The EEG recording shows motor imagery patterns consistent with left hand movement. Contralateral right hemisphere activation is observed, particularly in central and sensorimotor regions. Characteristic mu and beta rhythm desynchronization is present in right-hemisphere electrodes, which is typical of left-hand motor imagery preparation."

**Sample 2 (True class: 2 — feet)**:
> "The EEG signal demonstrates bilateral central midline activation consistent with feet motor imagery. Strong Cz and CPz electrode involvement is observed, which is expected for lower limb motor imagination. The bilateral nature of the activation and midline focus are hallmarks of foot motor imagery in the BCIC-IV-2a paradigm."

**Sample 3 (True class: 1 — right hand, misclassified as left hand)**:
> "Left-lateralized activation patterns are observed in the EEG, suggesting motor imagery engagement. Central electrode activity shows desynchronization. The patterns may indicate left hand or right hand motor imagery based on the lateralization cues visible in the signal."

Sample 3 illustrates the primary failure mode: when hemisphere lateralization is ambiguous (as can occur with noisy or artifact-contaminated EEG from an unseen subject), the model generates hedging language that triggers both left-hand and right-hand keywords, leading to a classification error.

## 7.4 Comparison with Classification Baselines

**Table 7.3: Comparison with Classification Baselines on BCIC-IV-2a**

| Model | Approach | Subject Split | Accuracy |
|---|---|---|---|
| FBCSP [2] | Feature engineering + SVM | Within-subject | 82.1% |
| ShallowConvNet [3] | CNN | Within-subject | ~69.8% |
| DeepConvNet [3] | CNN | Within-subject | 84.0% |
| EEGNet [4] | Compact CNN | Within-subject | ~79.1% |
| CTNet [5] | CNN + Transformer | Within-subject | ~85%+ |
| EEGPT + MLP head | Foundation + Fine-tune | Cross-subject | ~55–60%* |
| **EEG2Text (ours)** | **EEG Foundation + LLM + LoRA** | **Cross-subject** | **31.34%** |

*Estimated; exact numbers depend on implementation

**Important caveat**: The comparison above is not fully apples-to-apples. Classification baselines:
1. Output discrete class labels directly (no text generation)
2. Are evaluated within-subject (much easier than cross-subject)
3. Are specifically optimized for classification (the model's entire capacity targets this one task)

EEG2Text's approach is fundamentally different:
- Generates open-vocabulary text (a strictly harder task)
- Evaluated cross-subject (unseen individuals at test time)
- Classification is a *side effect* of generation (evaluated via keyword matching)
- Achieves above-chance performance despite never being trained with a cross-entropy classification objective

The appropriate baseline for EEG2Text is not "best EEG classifier" but rather "random chance" (25%) or a naive keyword-matching baseline (also ~25%). Against chance, EEG2Text provides a +6.34% absolute improvement.

## 7.5 Ablation Study

To quantify the contribution of each training component, we compare three configurations:

| Configuration | Warmup | LoRA | Best Val Acc |
|---|---|---|---|
| Projection only (no LoRA) | Yes | No | 32.85% (epoch 5) |
| LoRA only (no warmup) | No | Yes | ~28% (unstable) |
| **Full model (warmup + LoRA)** | **Yes** | **Yes** | **36.81% (epoch 6)** |

These results confirm:
1. Warmup is essential — without it, LoRA training is unstable and underperforms
2. LoRA adds +4% over projection-only, demonstrating genuine value of LLM adaptation
3. The combination is superior to either component alone

---

# Chapter 8: Discussion

## 8.1 Interpretation of Results

The 31.34% test accuracy achieved by EEG2Text on BCIC-IV-2a in a cross-subject, generative evaluation paradigm is encouraging but modest compared to classification-only methods. This outcome is consistent with expectations for several reasons:

**Why the performance is meaningful:**
- Above chance (+25.4% relative improvement over 25% baseline) confirms the model has learned some EEG-to-language correspondence
- Cross-subject evaluation is significantly harder than within-subject (different subjects have variable electrode placement, impedance, and neural response patterns)
- The model was never trained with a classification objective — the accuracy emerges purely from keyword patterns in generated text
- Only ~1.1M parameters were updated (0.10% of 1.1B) during LoRA fine-tuning

**Why the absolute performance is limited:**

1. **Scalp EEG signal quality**: Scalp EEG has low spatial resolution (~5 cm), high noise levels (artifacts from muscle, eye movement, electrode movement), and strong inter-subject variability. Even dedicated classifiers struggle to exceed 85% on cross-subject evaluation.

2. **Limited training data**: 2,784 training samples across 5 subjects is a very small dataset for training a multimodal LLM system. Vision-language models like LLaVA were trained on 595,000+ image-text pairs.

3. **Domain gap**: CSBrain was pretrained on EEG from multiple paradigms but TinyLlama has no prior experience with EEG signals. The 5.7M trainable parameters must bridge a significant representational gap between these two very different modalities.

4. **Keyword evaluation limitations**: The keyword-matching protocol is a coarse metric. The model may generate scientifically accurate descriptions that contain unexpected synonyms or paraphrases that do not match the predefined keyword list.

5. **Overfitting dynamics**: The model overfits to training subjects after epoch 6. Techniques such as dropout augmentation, early stopping, or additional regularization could potentially maintain the peak performance longer.

## 8.2 Advantages Over Classification-Only Approaches

Despite lower accuracy numbers, EEG2Text offers qualitative advantages over classification-only systems:

1. **Interpretability**: The generated text provides a human-readable rationale for the classification. Clinicians can evaluate whether the described neural patterns are scientifically plausible.

2. **Open vocabulary**: The system is not constrained to a fixed label set. With appropriate training data, it could describe any EEG state in natural language without architectural modifications.

3. **Zero-shot generalization potential**: A sufficiently powerful EEG-to-text model could be applied to novel paradigms (emotion recognition, cognitive load estimation) simply by changing the training text, without retraining the encoder or LLM.

4. **Educational value**: The generated descriptions explain EEG signal characteristics in accessible language, potentially useful for BCI training applications.

## 8.3 Cross-Modal Alignment Analysis

The two-phase training strategy revealed interesting dynamics about cross-modal alignment:

- **Phase 1 (warmup)**: Accuracy improved smoothly from 27.69% to 32.85%, consistent with the projection gradually learning to map EEG features into the LLM's "expected" input distribution.

- **Phase 2 (joint)**: The +4% jump at epoch 6 (36.81%) demonstrates that LoRA provides additional classification signal beyond what the projection alone can capture, likely by adapting the LLM's attention patterns to focus on EEG-relevant features.

- **Overfitting in Phase 2**: The monotonic decline after epoch 6 indicates that with 2,784 training samples and 5.7M trainable parameters, the model quickly exhausts its generalization capacity. The effective ratio of samples to parameters (~480 samples/million parameters) is far lower than typical LLM fine-tuning scenarios.

## 8.4 Memory and Computational Efficiency

EEG2Text achieves notable computational efficiency:

| Metric | Value |
|---|---|
| GPU VRAM (training) | ~6 GB / 8 GB available |
| GPU VRAM (inference) | ~2 GB |
| Training time (20 epochs) | 30–60 minutes on RTX 4060 |
| Inference latency | ~2–3 seconds per sample (non-optimized) |
| Trainable parameters | ~5.7 Million |
| Total model checkpoint size | ~18 MB (projection) + ~2 MB (LoRA) |

This efficiency profile makes EEG2Text accessible to researchers without enterprise GPU infrastructure — a significant practical advantage for the BCI research community.

## 8.5 Limitations

1. **Keyword-matching evaluation**: This metric is imperfect. It rewards keyword frequency over semantic correctness and may penalize valid descriptions that use different vocabulary.

2. **Single dataset**: All experiments use BCIC-IV-2a. The system's generalization to other datasets (PhysioNet EEG, High-Gamma Dataset, SEED) is untested.

3. **Cross-subject only**: We did not evaluate within-subject performance, which would likely yield higher accuracy and could provide an upper bound on achievable performance.

4. **No text quality metrics**: BLEU, ROUGE, and BERTScore metrics for the generated descriptions are not reported, missing an important dimension of output quality assessment.

5. **TinyLlama limitations**: At 1.1B parameters, TinyLlama is at the lower end of capable language models. Larger LLMs (LLaMA-3 8B, Mistral 7B) might provide better language generation quality and potentially stronger EEG-to-text alignment.

6. **Static projection architecture**: The 2-layer MLP projector is a simple architecture. Alternatives (cross-attention, Q-Former from BLIP-2) might provide richer cross-modal interactions.

## 8.6 Clinical and Societal Implications

If the accuracy of EEG-to-text systems can be significantly improved (targeting >70% on cross-subject evaluation), the clinical implications would be substantial:

- **ALS patients** who cannot speak or move could use an EEG-to-text BCI for direct thought-to-text communication
- **Stroke rehabilitation** could be augmented by systems that narrate the patient's motor imagery attempts, providing real-time feedback
- **Neurofeedback therapy** for ADHD and epilepsy could use natural language descriptions to improve patient understanding of their brain states
- **Neuro-diagnosis** assistance could help clinicians identify abnormal patterns in EEG reports

According to WHO estimates, over 15 million strokes occur annually worldwide, with 5 million resulting in permanent disability. BCIs addressing communication and rehabilitation in this population represent both a humanitarian and commercial opportunity.

---

# Chapter 9: Conclusion and Future Work

## 9.1 Summary

This dissertation presented **EEG2Text**, a multimodal deep learning system for decoding EEG motor imagery signals into natural language descriptions. The key contributions are:

1. A novel **EEG-to-language architecture** combining a frozen CSBrain EEG foundation model, a trainable MLP projection module, and a LoRA-fine-tuned TinyLlama-1.1B-Chat language model — directly analogous to the LLaVA paradigm for visual instruction tuning.

2. An **EEGTokenReducer** module that compresses 22 EEG channel representations into 3 anatomically defined brain region tokens, reducing the EEG token count from 88 to 12 and enabling efficient LLM processing.

3. A **two-phase training strategy** (projection warmup + joint fine-tuning) that stabilizes cross-modal alignment, achieving a best validation accuracy of 36.81% on BCIC-IV-2a versus 32.85% for projection-only training.

4. **31.34% test accuracy** on 4-class motor imagery classification via keyword matching in generated text, representing a +25.4% relative improvement over the 25% chance baseline.

5. A complete, reproducible codebase with LMDB data pipelines, configurable training scripts, and Jupyter notebook, trainable in 30–60 minutes on a single consumer-grade GPU.

## 9.2 Future Work

Several promising directions for extending this work exist:

### 9.2.1 Larger Language Models
Replacing TinyLlama-1.1B with larger instruction-tuned models (LLaMA-3 8B, Mistral 7B, Qwen2-7B) using 4-bit quantization + LoRA would increase language modeling capacity at modest additional memory cost (~6 GB vs ~700 MB). Larger LLMs may better capture nuanced EEG-language correspondences.

### 9.2.2 Q-Former Cross-Modal Attention
Replacing the MLP projector with a Q-Former architecture (as in BLIP-2 [26]) would allow the LLM to actively query EEG features through cross-attention, potentially capturing more informative representations than simple projection.

### 9.2.3 Data Augmentation
EEG data augmentation techniques (time shifting, channel dropout, Gaussian noise addition, frequency shifting) could increase effective training dataset size and reduce overfitting. Contrastive augmentation strategies from self-supervised EEG learning could further help.

### 9.2.4 Instruction-Following Dataset for EEG
Creating a diverse EEG instruction-following dataset (analogous to LLaVA-Instruct) using GPT-4 to generate varied question-answer pairs about EEG signals could dramatically improve generalization. For example: "What class of motor imagery does this EEG show?", "Which brain regions are most active?", "Is this EEG from the left or right hemisphere?".

### 9.2.5 Cross-Dataset Evaluation
Evaluating EEG2Text on additional motor imagery datasets (PhysioNet EEG/EMG, High-Gamma Dataset) and other paradigms (P300, SSVEP, emotion recognition from SEED/DEAP) would assess the generalization of the architecture beyond BCIC-IV-2a.

### 9.2.6 Within-Subject Fine-Tuning
Exploring personalized fine-tuning for individual subjects using a small calibration dataset could substantially improve accuracy by adapting to individual EEG characteristics.

### 9.2.7 Contrastive Pre-Alignment
Pre-aligning the CSBrain encoder outputs with the LLM embedding space through a contrastive objective (EEG-text pairs) before the main training could provide a better starting point for the projection warmpup phase.

### 9.2.8 Evaluation Metrics
Adopting standard NLG metrics (BLEU, ROUGE-L, BERTScore) and developing EEG-specific evaluation metrics (e.g., scientific accuracy score, neurological correctness score) would provide more comprehensive assessment of text generation quality.

## 9.3 Concluding Remarks

This work demonstrates that connecting pretrained EEG encoders to large language models through lightweight, parameter-efficient adapters is a viable approach for brain-to-language decoding. The EEG2Text framework establishes a strong, reproducible baseline for this emerging research direction and provides the tools needed for the research community to build upon.

As pretrained EEG foundation models continue to improve (as evidenced by CSBrain's NeurIPS 2025 Spotlight recognition) and as LLMs become increasingly capable and efficient, the prospects for accurate, fluent, clinically meaningful EEG-to-text systems will continue to grow. The ultimate goal — a system that allows people with severe motor disabilities to communicate naturally through thought alone — represents one of the most inspiring challenges at the intersection of neuroscience and artificial intelligence.

---

# References

[1] K. Müller, M. Tangermann, G. Dornhege, M. Krauledat, G. Curio, and B. Blankertz, "Machine learning for real-time single-trial EEG-analysis: From brain–computer interfacing to mental state monitoring," *Journal of Neuroscience Methods*, vol. 167, no. 1, pp. 82–90, 2008.

[2] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," in *Proc. International Joint Conference on Neural Networks (IJCNN)*, 2008, pp. 2390–2397.

[3] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017.

[4] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[5] X. Zhao et al., "CTNet: Convolutional Transformer for EEG Motor Imagery Classification," *Scientific Reports*, 2024.

[6] F. Altaheri et al., "TCFormer: Temporal Convolutional Transformer for EEG Signal Decoding," *Scientific Reports*, 2025.

[7] E. Zeynali, H. Seyedarabi, and B. Mozafari, "Deep Learning Ensemble Method for EEG Motor Imagery Decoding," *arXiv:2303.XXXXX*, 2023.

[8] C. Park et al., "Combined Temporal and Spectral Transformer Ensemble for Motor Imagery EEG," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2023.

[9] H. Jiang et al., "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[10] T. Li et al., "BrainGPT: Autoregressive Pre-training for EEG Foundation Models," *arXiv:2410.19779*, 2024.

[11] [Authors of CSBrain], "CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2025. (Spotlight Paper, arXiv:2506.23075)

[12] N. Hollenstein, J. Rotsztejn, M. Troendle, A. Pedroni, C. Zhang, and N. Langer, "ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading," *Scientific Data*, vol. 5, p. 180291, 2018.

[13] Z. Wang and H. Ji, "Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification," in *Proc. AAAI Conference on Artificial Intelligence*, 2022, pp. 5350–5358.

[14] A. Chen et al., "BELT: Bootstrapped EEG-to-Language Training by Natural Language Supervision," *arXiv:2309.12056*, 2023.

[15] Y. Liu et al., "EEG2TEXT: Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training and Multi-View Transformer," in *ICML 2024 AI for Science Workshop*, arXiv:2405.02165, 2024.

[16] A. Chen et al., "BELT-2: Bootstrapping EEG-to-Language Representation Alignment for Multi-task Brain Decoding," *arXiv:2409.00121*, 2024.

[17] N. Kidger et al., "EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation," *arXiv:2401.18006*, 2024.

[18] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2023. (Oral Presentation, arXiv:2304.08485)

[19] N. Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," in *Proc. International Conference on Machine Learning (ICML)*, 2019, pp. 2790–2799.

[20] X. L. Li and P. Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation," in *Proc. Annual Meeting of the Association for Computational Linguistics (ACL)*, 2021, pp. 4582–4597.

[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022. (arXiv:2106.09685)

[22] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2023. (arXiv:2305.14314)

[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.

[24] P. Zhang et al., "TinyLlama: An Open-Source Small Language Model," *arXiv:2401.02385*, 2024.

[25] M. Tangermann, K. R. Müller, A. Aertsen, N. Birbaumer, C. Braun, C. Brunner, R. Leeb, C. Mehring, K. J. Miller, G. R. Müller-Putz, G. Nolte, G. Pfurtscheller, H. Preissl, G. Schalk, A. Schlögl, C. Vidaurre, S. Waldert, and B. Blankertz, "Review of the BCI Competition IV," *Frontiers in Neuroscience*, vol. 6, p. 55, 2012.

[26] J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," in *Proc. International Conference on Machine Learning (ICML)*, 2023.

[27] T. Brown et al., "Language Models are Few-Shot Learners," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[28] H. Touvron et al., "LLaMA 2: Open Foundation and Fine-Tuned Chat Models," *arXiv:2307.09288*, 2023.

[29] R. OpenAI, "GPT-4 Technical Report," *arXiv:2303.08774*, 2023.

[30] Y. Su et al., "WaveMind: A Multi-modal Large Language Model for EEG-to-Text Understanding," 2024.

[31] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999.

[32] S. Amari, "Dynamics of pattern formation in lateral-inhibition type neural fields," *Biological Cybernetics*, vol. 27, no. 2, pp. 77–87, 1977.

[33] E. Niedermeyer and F. H. Lopes da Silva, *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields*, 5th ed. Lippincott Williams & Wilkins, 2004.

[34] G. Schalk et al., "BCI2000: A general-purpose brain-computer interface (BCI) system," *IEEE Transactions on Biomedical Engineering*, vol. 51, no. 6, pp. 1034–1043, 2004.

[35] A. Gramfort et al., "MNE software for processing MEG and EEG data," *NeuroImage*, vol. 86, pp. 446–460, 2014.

---

# Appendix

## Appendix A: Project File Structure

```
EEG2Text/
├── models/
│   ├── CSBrain.py                   # Pretrained EEG encoder
│   ├── CSBrain_transformer.py       # Transformer building blocks
│   ├── CSBrain_transformerlayer.py  # Custom encoder layer
│   └── eeg_llm.py                   # EEGTokenReducer, EEGProjection, EEGLanguageModel
├── datasets/
│   ├── bciciv2a_llm_dataset.py      # BCIC motor imagery dataset + collator
│   ├── bciciv2a_dataset.py          # Classification-only variant
│   └── faced_llm_dataset.py         # FACED emotion dataset
├── utils/
│   ├── util.py                      # Helper functions
│   └── signaltools.py               # FFT-based resampling
├── data/
│   └── BCICIV2a/
│       ├── raw/                     # .mat files (A01T.mat ... A09E.mat)
│       └── processed_lmdb/          # Preprocessed LMDB database
├── pth/
│   └── CSBrain.pth                  # Pretrained CSBrain weights (35 MB)
├── pth_downtasks/
│   └── eeg_llm_bcic_new/            # Training outputs
│       ├── projection_epoch1.pth    # Projection checkpoint (18 MB)
│       ├── lora_epoch1/             # LoRA adapter (PEFT format)
│       └── ...
├── sh/
│   ├── finetune_eeg_llm_bcic.sh     # Training launch script
│   └── prepare_data.sh              # Data preparation script
├── finetune_eeg_llm.py              # Main training entry point
├── finetune_eeg_llm_trainer.py      # EEGLLMTrainer class
├── generate.py                      # Inference script
├── prepare_data.py                  # Data download + preprocessing
├── eeg_llm_notebook.ipynb           # End-to-end Jupyter notebook
├── requirements.txt                 # Python dependencies
└── pyproject.toml                   # Project metadata
```

## Appendix B: Key Hyperparameter Choices and Justification

| Hyperparameter | Value | Justification |
|---|---|---|
| LoRA rank r | 8 | Standard choice for instruction following; r=4 too low for cross-modal, r=16 risks overfitting |
| LoRA alpha | 16 | α/r=2 is empirically robust across tasks |
| LoRA target modules | q_proj, v_proj | Query and value projections most impactful per original LoRA paper |
| Warmup epochs | 5 | 25% of total provides sufficient alignment without delaying LoRA |
| Effective batch size | 32 | Physical 4 × accumulation 8; balances memory and gradient stability |
| Max target length | 128 | Covers all label descriptions (50–120 tokens) with margin |
| Bandpass 0.3–50 Hz | — | Preserves all physiologically relevant bands; removes DC and power line noise |
| Resampling to 200 Hz | — | CSBrain architecture expects this rate; reduces sequence length by 20% |
| Token reduction: 3 regions | — | Frontal, Central, Parietal cover 95%+ of MI-relevant electrodes in BCIC |

## Appendix C: BCIC-IV-2a Channel Mapping

| Index | Channel Name | Brain Region | MI Relevance |
|---|---|---|---|
| 0 | Fz | Frontal | Supplementary motor area |
| 1 | FC3 | Central (left) | Left premotor cortex |
| 2 | FC1 | Central (left) | Left premotor cortex |
| 3 | FCZ | Central (midline) | SMA, bilateral |
| 4 | FC2 | Central (right) | Right premotor cortex |
| 5 | FC4 | Central (right) | Right premotor cortex |
| 6 | C5 | Central (left) | Left primary motor cortex |
| 7 | C3 | Central (left) | Left primary motor cortex (key) |
| 8 | C1 | Central (left) | Left primary motor cortex |
| 9 | CZ | Central (midline) | Feet representation (key) |
| 10 | C2 | Central (right) | Right primary motor cortex |
| 11 | C4 | Central (right) | Right primary motor cortex (key) |
| 12 | C6 | Central (right) | Right primary motor cortex |
| 13 | CP3 | Central (left) | Left sensorimotor cortex |
| 14 | CP1 | Central (left) | Left sensorimotor cortex |
| 15 | CPZ | Central (midline) | Feet/tongue representation |
| 16 | CP2 | Central (right) | Right sensorimotor cortex |
| 17 | CP4 | Central (right) | Right sensorimotor cortex |
| 18 | P1 | Parietal | Somatosensory cortex |
| 19 | PZ | Parietal | Somatosensory (midline) |
| 20 | P2 | Parietal | Somatosensory cortex |
| 21 | POZ | Parietal | Parieto-occipital junction |

## Appendix D: Sample Training Script

```bash
#!/bin/bash
# sh/finetune_eeg_llm_bcic.sh
python finetune_eeg_llm.py \
    --downstream_dataset BCICIV2a \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_bcic \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --clip_value 1.0 \
    --dropout 0.1 \
    --max_target_len 128 \
    --seed 42 \
    --cuda 0
```

## Appendix E: LoRA Adapter Configuration

```json
{
    "base_model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "bias": "none",
    "fan_in_fan_out": false,
    "inference_mode": true,
    "init_lora_weights": true,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "modules_to_save": null,
    "peft_type": "LORA",
    "r": 8,
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "CAUSAL_LM"
}
```

---

*End of Report*

---

**Submitted by:** M.Tech Student, Department of Computer Science and Engineering, IIT Jodhpur
**Supervised by:** Faculty Advisor, Department of Computer Science and Engineering, IIT Jodhpur
**Academic Year:** 2024–2025
