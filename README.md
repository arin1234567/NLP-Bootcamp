# README — Mini BERT (from-scratch) — Submission

## Overview
This notebook implements a mini BERT encoder (Transformer encoder stack) from scratch in PyTorch and trains it jointly on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) using WikiText-2 dataset.

## Files to submit
- `mini_bert_notebook.ipynb` (this Colab notebook)
- `mini_bert_checkpoint.pt` (optional small checkpoint)
- `README.md` (explain model hyperparameters and results)

## Model details
- Layers: 4 encoder layers
- Hidden size: 256
- Heads: 4
- FFN dim: 512
- Max seq len: 128

## Training
- Dataset: WikiText-2 (wikitext-2-v1)
- Objectives: MLM (15% mask) + NSP (binary)
- Tokenizer: `bert-base-uncased` (HuggingFace; tokenizer only)

## How to run
1. Open in Google Colab.
2. Make sure GPU runtime is selected.
3. Run cells top-to-bottom.

## Results (example)
- Final validation MLM accuracy and NSP accuracy printed at the end of the run.
- Sample masked predictions printed for qualitative check.

## Notes and improvements
- Increase `num_epochs`, `batch_size`, and `hidden_size` for better performance (requires more GPU memory/time).
- Could add more sophisticated sentence splitting and negative sampling strategies for NSP.
- Consider longer pretraining or using larger corpus for stronger representations.
