# Vision Transformer (ViT)

The Vision Transformer (ViT) is a deep learning model that uses transformer-based architectures, originally created for natural language processing, to handle computer vision tasks like image classification.---


## HuggingFace Model

Model: [ViT from Scratch](https://huggingface.co/Parthiban007/VisionTransformer)

## Architecture
The Vision Transformer (ViT) reimagines image classification by applying the standard transformer architecture—originally used for NLP—to image data. Here's how it works:
### 1. Image to Patches
- The input image is split into non-overlapping patches (e.g., 16×16 pixels).
- Each patch is flattened and projected into a vector using a trainable linear layer.
- This results in a sequence of patch embeddings, similar to token embeddings in NLP.
### 2. Patch + Positional Embedding
- A special [CLS] token is prepended to the sequence. It will represent the whole image during classification.
- Positional embeddings are added to preserve spatial information (since transformers are permutation-invariant).
### 3. Transformer Encoder
- The embedded sequence is passed through a stack of standard transformer encoder layers.
- Each encoder layer contains:
- Multi-head Self-Attention (MSA)
- Layer Normalization (LN)
- Feedforward MLP block
- Residual connections
### 4. Classification Head
- The output corresponding to the [CLS] token is passed through an MLP head to produce the final classification logits.

---

##  Features
- Transformer-based vision model
- Flexible patch size and image resolution
- Pretrained model support
- Easy training and evaluation scripts
- Configurable model sizes (e.g., ViT-B/16, ViT-L/16)

---
## Vison Transformer(ViT):
<img src="https://miro.medium.com/max/1200/1*rGi2u0IUNhQSm6CC8z2mdg.png">


## Acknowledgements

> 📄 **Paper**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- **YT Video**: [Vizuara ViT](https://youtu.be/DdsVwTodycw?si=QB_yRUmPcKK-R69e)
