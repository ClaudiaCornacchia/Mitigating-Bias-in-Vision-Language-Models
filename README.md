# Mitigating Bias in Vision-Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ClaudiaCornacchia/CV_Project/blob/main/CV_Project_Claudia_Cornacchia_1986434.ipynb)

**Author:** Claudia Cornacchia   

## Overview
This project addresses intersectional bias (for **gender** and **ethnicity**) in Vision-Language Models, specifically focusing on **CLIP**. I investigate and compare two mitigation strategies to reduce stereotyping in a **predefined set of professions** without degrading zero-shot generalization:   
- **Fine-Tuning** (LoRA): training on a custom, perfectly balanced dataset (DebiasingDB) to force the model to unlearn correlations between demographic groups and professions.  
- **Inference-Time Ablation**: a training-free geometric approach that identifies and projects out the bias subspace from the model's embeddings.
 
The results demonstrate that Inference-Time Ablation is the superior method, it effectively blinds the model to protected attributes, achieving near zero representation bias, and good classification and retrieval fairness, without compromising the model’s pre-trained knowledge.    

> **Note:** Consult the [project notebook](./CV_Project_Claudia_Cornacchia_1986434.ipynb) for detailed documentation on the used datasets, evaluation strategy, and debiasing logic.   

##  Project Structure
### 1. Data and Setup
- Initialization: setup of global variables and loading of pre-trained CLIP models (ViT-B/32, L/14, H/14).
- DebiasingDB: creation, augmentation, and splitting (fine-tuning and validation) of the custom balanced dataset.
- Evaluation datasets: FairFace (for occupation-bias evaluation) and Oxford-Pets/Food-101 (for zero-shot robustness).

### 2. Evaluation Framework
- Implementation of the three key metrics:
  - Representation Bias (RB): measuring intrinsic priors using Gaussian noise.
  - Occupation Bias: measuring stereotyping (classification) and exclusion (retrieval).
  - Zero-Shot Generalization: ensuring no catastrophic forgetting occurs.
    
### 3. Mitigation Strategy I: Fine-Tuning (LoRA)
- Implementation of parameter-efficient fine-tuning using Low-Rank Adaptation.
- Two experiments: Vision-Only (freezing text) vs. Joint Vision-Text optimization.
- Analysis of training loss and evaluation metrics.

### 4. Mitigation Strategy II: Inference-Time Ablation
- Subspace Projection: calculating and removing bias directions geometrically.
- Validation: testing the strategy's robustness on an unseen dataset (UTKFace).
- Visualization: Using t-SNE to visualize the embedding space before and after debiasing.
- Evaluation metrics.

### 5. Ablation Study 
- Analysis of different architectures (ViT-L/14) and data scales (LAION-2B / ViT-H/14) to understand how model size and training data affect intersectional bias.

## Requirements
```bash
pip install torch torchvision transformers peft datasets pandas numpy scipy scikit-learn matplotlib seaborn tqdm
```

## References
[1] Ibrahim Alabdulmohsin, et al. *CLIP the Bias: How Useful is Balancing Data in Multimodal Learning?* ICLR 2024

[2] M. D'Incà, et al. *OpenBias: Open-set Bias Detection in Text-to-Image Generative Models.* CVPR 2024

[3] Neale Ratzlaff, et al. *Debiasing Large Vision-Language Models by Ablating Protected Attribute Representations*. NeurIPS workshop on SafeGenAl 2024

[4] Md Mehrab Tanjim, et al. *Discovering and Mitigating Biases in CLIP-based Image Editing*. WACV 2024

[5] Tolga Bolukbasi, et al. *Man is to computer programmer as woman is to homemaker? Debiasing word embeddings*. Advances in neural information processing systems, 29, 2016

[6] Thomas Manzini, et al. *Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings*. Association for Computational Linguistics, 2019

[7] Alec Radford, et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021

[8] https://huggingface.co/datasets/NYUAD-ComNets/

[9] Nouar AlDahoul, et al. *AI-generated faces influence gender stereotypes and racial homogenization*. 2024

[10] Boyu Han, et al. *LightFair: Towards an Efficient Alternative for Fair T2I Diffusion via Debiasing Pre-trained Text Encoders*. NeurIPS 2025

