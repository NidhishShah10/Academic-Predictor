# 🎓 Academic Predictor

Academic Predictor is a machine learning project that predicts whether a student is academically **At Risk** or **Satisfactory** based on academic, personal, and environmental factors.

The project includes a locally running Streamlit application where users can enter student information and receive predictions.

## Project Overview

The system analyzes 19 student features, including:
- Hours studied
- Attendance
- Previous exam scores
- Motivation level
- Parental involvement
- Access to educational resources
- Family income
- Other academic and environmental factors

The goal is to explore how machine learning can help identify students who may need additional academic support.

## Machine Learning Models

The following algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

**Best Model:** Logistic Regression

### Final Test Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.59% |
| Precision | 96.98% |
| Recall | 96.02% |
| F1-Score | 96.50% |
| ROC-AUC | 0.9994 |

These results were obtained using the Student Performance Factors dataset and may not reflect performance on real-world student populations.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Dataset

**Student Performance Factors Dataset**

Source: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

The dataset contains 6,607 student records.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/NidhishShah10/Academic-Predictor.git
   cd Academic-Predictor
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python academic_predictor.py
   ```

4. Launch the Streamlit application:

   ```bash
   python -m streamlit run app.py
   ```

The application will open locally in your browser, usually at `http://localhost:8501`.

## Project Team

- Nidhish Shah
- Kyle Do

Oakland University — Machine Learning (CSI 4170)

## Disclaimer

This project is intended for educational and research purposes only. Predictions are based on a machine learning model and should not be considered official academic assessments.