# AI-Trust Forensic Suite

**Interactive Web Application:** [https://datscapstoneproject.streamlit.app/](https://datscapstoneproject.streamlit.app/)

This repository contains the code and model weights for a multi-task deep learning forensic tool designed to detect AI-generated images and attribute them to their specific source generator. The project was developed as a Capstone for DATS 6499 at George Washington University.

## Overview

As generative models like Midjourney and DALL·E 3 reach photographic parity with reality, the risk of visual misinformation has scaled drastically. Current commercial detection tools (e.g., Undetectable.ai, Sightengine) operate as "black boxes"—they output a binary percentage score with no explanation of *why* an image was flagged or *which* model generated it. 

This project addresses that gap by building a forensic system that:
1. **Detects** whether an image is Authentic (Real) or AI-Generated.
2. **Attributes** AI-generated images to one of five source models (Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3, or Midjourney).
3. **Explains** its reasoning visually using Attention Rollout and Gradient Saliency Maps.

## Architecture

We use a **Multi-Task Vision Transformer (ViT-Base/16)** to process images. 
Unlike Convolutional Neural Networks (CNNs) which focus on local pixel artifacts, the self-attention mechanism of the ViT looks at the entire image globally. This is crucial because modern diffusion models leave statistical artifacts distributed across the entire image rather than in localized pockets.

The system utilizes transfer learning via an ImageNet-21k pre-trained ViT. We freeze the first 6 layers to retain generic visual features and fine-tune the final 6 layers specifically for forensic artifact detection.

The shared ViT backbone outputs a standard `[CLS]` token, which is then fed simultaneously into two separate classification heads:
- **Head A (Binary):** Real vs. Synthetic.
- **Head B (Attribution):** Real, SD 2.1, SDXL, SD 3, DALL·E 3, Midjourney.

## Addressing Data Imbalance and Overfitting

The project was trained on a 96,000-image subset of the Defactify dataset. During the exploratory data analysis, we identified a critical 1:5 class imbalance between Authentic images (~16.7%) and AI-Generated images (~83.3%). 

To prevent the model from blindly predicting "AI-Generated" to achieve high baseline accuracy, we iterated through several training configurations:
- **Class Weighting:** We applied an `[8.0, 1.0]` weight penalty to the Cross-Entropy loss in Head A, heavily penalizing the misclassification of Authentic photos.
- **Task Prioritization:** We weighted the overall loss function to prioritize Head A (Detection) over Head B (Attribution) via an `alpha=2.0` multiplier.
- **Regularization:** To eliminate severe initial overfitting (where train loss dropped to `0.10` but validation loss stalled at `0.34`), we introduced **L1 regularization (1e-5)**, increased **L2 weight decay to 0.05**, and added **label smoothing (0.1)**. 

These adjustments resulted in clean convergence over 15 epochs, demonstrating strong generalization without memorizing the dataset.

## Explainable AI (XAI) Integration

To provide transparency for end-users, we integrated two XAI techniques using the `captum` library:
- **Attention Rollout:** Aggregates the self-attention weights across all 12 layers of the ViT, producing a heatmap that highlights which regions of the image the model focused on (e.g., unnatural lighting or distorted hands).
- **Gradient Saliency:** Computes pixel-level gradients to show precisely which pixels had the highest positive impact on the model's final prediction.

## Repository Structure

```text
├── app/
│   └── streamlit_app.py      # The Streamlit web dashboard
├── checkpoints/
│   └── best_model.pt         # Fine-tuned PyTorch model weights (managed via Git LFS)
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis for visualizations
│   └── train_colab.ipynb     # The Colab training pipeline
├── src/
│   ├── dataset.py            # HuggingFace dataset logic and dataloaders
│   ├── evaluate.py           # Confusion matrices and classification reports
│   ├── explainability.py     # Captum XAI heatmap generation
│   ├── model.py              # Multi-task ViT backbone and classification heads
│   ├── train.py              # Training loop with custom weighted loss and L1/L2 regularization
│   └── utils.py              # Hyperparameters and transforms
├── .gitattributes            # Git LFS configuration for .pt files
└── requirements.txt          # Minimal dependencies for Streamlit production deployment
```

## Local Installation

If you prefer to run the Streamlit app locally instead of using the live link:

1. Clone the repository including the large `.pt` file:
   ```bash
   git clone https://github.com/hussainnathani/DATS_Capstone_Project.git
   ```
2. Install the necessary packages. (Note: standard PyTorch is pre-supposed on your system):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Author
Mohammed Hussain Nathani (G36308827)
George Washington University — DATS 6499 Applied Research
