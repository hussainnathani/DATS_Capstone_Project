# DATS_Capstone_Project
AI-Trust Forensic Suite is an end-to-end deep learning + explainable AI web app that detects whether an image is Authentic vs. AI-generated and attributes the likely AI generator. It addresses the “forensic transparency gap” by pairing predictions with visual evidence attention heatmaps in a simple drag-and-drop Streamlit dashboard.

=========================================

1) Problem Addressed
--------------------
AI-generated images (deepfakes / diffusion outputs) are increasingly realistic. This creates a verification gap:
most people cannot reliably tell whether an image is real or AI-generated, which increases misinformation,
fraud, and trust issues. Many tools give only a “real vs fake” score without explaining WHY.

2) Proposed Solution
--------------------
This project builds an AI forensic demo that:
- Classifies an input image as: Authentic (Real) vs Synthetic (AI-generated)
- Produces an explainability heatmap (saliency / attention-based visualization) showing which regions
  influenced the prediction.
The key goal is not only prediction accuracy, but also interpretability (“forensic transparency”).

3) Tech Stack Used
------------------
- Python 3.x
- Deep Learning: PyTorch
- Vision model backbone: Vision Transformer (ViT) or CNN (configurable)
- Explainability: Captum (or gradient/saliency implementation)
- Web demo: Streamlit (simple drag-and-drop upload UI)
- Supporting: NumPy, Pandas, OpenCV/Pillow, Matplotlib

4) Repository Contents
----------------------
- requirements.txt   -> Python dependencies (with versions)
- src/               -> Source code (training, inference, explainability, utilities)
- app/ (optional)    -> Streamlit app entrypoint (demo UI)
- docs/ (optional)   -> proposal/report/slides and architecture diagram
- results/ (optional)-> metrics, plots, sample outputs

Note: Large datasets and trained weights are not committed to GitHub by default.

5) Steps to Launch the Demo
---------------------------
A) Setup Environment
1. Create and activate a virtual environment:
   python -m venv .venv

   Windows:
   .venv\Scripts\activate
   Mac/Linux:
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

B) Run the Web App (Streamlit Demo)
1. Start the demo:
   streamlit run app/streamlit_app.py

2. Open the local URL shown in the terminal (usually http://localhost:8501)
3. Upload an image and view:
   - Prediction (Real vs AI)
   - Confidence score
   - Explainability heatmap output

6) Output
---------
- Predictions and heatmaps are saved in:
  results/figures/  and/or  results/predictions/
- Evaluation metrics (if run) are saved in:
  results/metrics/

Author: Mohammed Hussain Nathani
