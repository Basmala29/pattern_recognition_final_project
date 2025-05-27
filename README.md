 Mini GPT-2 PyTorch Implementation
==================================

Overview
--------

This repository contains a minimal PyTorch implementation of a GPT-2 style language model, trained on a custom dataset. It includes all core components such as positional encoding, multi-head self-attention, transformer blocks, and feed-forward layers.

The code demonstrates training, perplexity calculation, and text generation with the model.

Files Included
--------------

- gpt2_modell_weights.pth
  Pretrained weights of the Mini GPT-2 model saved in PyTorch format.

- FINAL REPORT PATTERN.pdf
  A comprehensive report template outlining the project's methodology, experiments, and results.

- final_project_pattern.ipynb
  A Google Colab notebook containing code and documentation for the final project following the report pattern.

Requirements
------------

- Python 3.7+
- PyTorch
- transformers
- tqdm
- matplotlib (optional for visualization)

How to Run
----------

1. Prepare your dataset and place it in the specified path.
2. Run the training script to train the model or load pretrained weights.
3. Generate text samples using the generation function.

Code Summary
------------

- PositionalEncoding adds sinusoidal positional information to token embeddings.
- MultiHeadSelfAttention implements scaled dot-product attention with multiple heads.
- TransformerBlock combines attention and feed-forward layers with layer normalization.
- GPT2 class stacks multiple transformer blocks to form the model.
- TextDataset processes raw text data into token sequences for training.
- Training loop uses AdamW optimizer and cross-entropy loss.
- Perplexity is computed to evaluate model performance.
- Text generation is done by autoregressively sampling tokens.
