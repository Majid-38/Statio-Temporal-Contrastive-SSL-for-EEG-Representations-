# CAST-CSSL EEG (Condition-Aware Spatio-Temporal Contrastive SSL)

This repository reproduces the code used in our paper on **CAST-CSSL**, a condition-aware spatio-temporal contrastive self-supervised learning framework for EEG-based resilience modelling under **subject-independent** evaluation.

## What CAST-CSSL does
- Learns **subject-invariant** but **stage/condition-discriminative** EEG embeddings from raw multichannel time series.
- Uses a mixed positive-pair strategy:
  - **Stage-aware pairing** (prob. `p=0.6`): windows from different recordings but same stage.
  - **Temporal-continuity pairing** (prob. `1-p=0.4`): adjacent non-overlapping windows from the same recording.
- Encoder: **Spatial 1×1 Conv** → **Residual temporal Conv** → **BiGRU** → **Self-attention** → **Pooling** → **MLP embedding head**.
- Projection head: 2-layer MLP (SimCLR-style), used only during SSL pretraining.
- Downstream: freeze encoder, average multiple crops per recording and per subject, train a **Logistic Regression probe**.

