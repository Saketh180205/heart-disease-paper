# 🫀 Heart Disease Prediction using Machine Learning

This repository contains a complete research project and Streamlit web app for predicting the likelihood of heart disease using machine learning algorithms.

## 📘 Overview
- **Dataset:** UCI Heart Disease dataset (303 samples)
- **Models Used:** Logistic Regression, Decision Tree, Random Forest, SVC, and an Ensemble Voting Classifier
- **Best Accuracy:** ~82% (Logistic Regression)
- **Frameworks:** Scikit-learn, Pandas, NumPy, Streamlit, Matplotlib, Seaborn

## 📊 Project Structure
heart-disease-paper/
│
├── data/                # dataset (heart.csv)
├── src/
│   ├── train.py         # trains and saves models
│   └── evaluate.py      # evaluates models and generates figures
│
├── figures/             # heatmaps, confusion matrices, ROC curves
├── results/             # model reports, saved models
├── streamlit_app.py     # Streamlit UI for prediction
└── README.md

## 🚀 Running Locally
1. Clone the repository:
   git clone https://github.com/Saketh180205/heart-disease-paper.git
   cd heart-disease-paper
2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
4. Add the dataset file:
   Place heart.csv inside the data/ folder.
5. Train and evaluate:
   python src/train.py
   python src/evaluate.py
6. Run the Streamlit app:
   streamlit run streamlit_app.py

## 📈 Results
- **Training Accuracy:** ~85%
- **Testing Accuracy:** ~82%
- Generated figures:
  - Correlation Heatmap  
  - Confusion Matrices  
  - ROC Curves  
  - Feature Importance (optional SHAP)

## 📜 Research Paper Integration
This project provides all analysis outputs required for a research paper:
- figures/ → visualizations (insert into report)
- results/ → accuracy metrics, classification reports, dataset summary

## 🌐 Deployment
You can deploy this Streamlit app on [Streamlit Cloud](https://share.streamlit.io) using your GitHub repository.

---
**Author:** Saketh V  
**Guided by:** [Your Professor / Supervisor Name]  
**Year:** 2025
