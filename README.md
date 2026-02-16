# MindVest — Behavioral Risk Prediction AI (Banking Proof-of-Concept)

MindVest is a machine learning–based behavioral risk prediction system designed to identify emotionally driven investment decisions during volatile market conditions.

This project was developed as a Proof-of-Concept (PoC) for a banking technology innovation challenge. It demonstrates how behavioral signals can be analyzed using machine learning to estimate emotional decision risk.

This system does NOT provide investment advice. Its purpose is to support behavioral awareness and responsible decision-making.

---

## Problem

During periods of market volatility, investors may make impulsive decisions influenced by emotional factors such as panic, fear, or FOMO (fear of missing out). These decisions can negatively affect both investors and financial institutions.

Financial systems currently evaluate financial suitability but often lack behavioral risk awareness at the moment of decision.

MindVest demonstrates how behavioral features can be used to estimate emotional decision risk using an interpretable machine learning approach.

---

## Solution

MindVest uses a machine learning pipeline to classify emotional decision risk based on behavioral signals.

The pipeline includes:

- Feature preprocessing using StandardScaler
- Logistic Regression model for interpretable classification
- Behavioral risk classification output: Low, Medium, or High risk
- Streamlit-based interactive demo interface

Logistic Regression was selected due to its interpretability, which is critical for financial and regulated environments.

---

## Dataset

The dataset used in this project is synthetic and was intentionally created to simulate behavioral investment decision scenarios.

Synthetic data allows demonstration of the machine learning pipeline and behavioral risk classification logic without relying on real user financial data. This ensures privacy safety while enabling realistic Proof-of-Concept development.

The dataset contains simulated behavioral features such as:

- Decision consistency
- Reaction speed
- Risk tolerance indicators
- Behavioral response patterns

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

---

## Project Structure

- `src/app.py` — Main application containing ML pipeline and Streamlit demo
- `data/mindvest_demo.csv` — Synthetic behavioral dataset used for the demo
- `docs/MindVest_sunum.pdf` — Project presentation

---

## How it works

1. Synthetic behavioral features are loaded from the dataset  
2. Data is preprocessed using a machine learning pipeline  
3. Logistic Regression model predicts emotional decision risk level  
4. Risk classification is displayed through the Streamlit interface  

---

## Purpose

This project demonstrates how behavioral machine learning models can be used in financial systems to support safer and more informed investment decision processes.

The goal is to show the feasibility of behavioral risk prediction in a regulated, privacy-safe, and interpretable machine learning framework.

---

## Author

Nevroz Aslan  
Software Engineering Student  
Machine Learning & Data Science Focus  
