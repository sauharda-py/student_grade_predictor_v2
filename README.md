# 🎓 Student Grade Predictor

An interactive **Streamlit** web app that predicts a student's final mathematics grade (G3) using Linear Regression trained on the UCI Student Performance dataset.

## Features

- 🎚️ **Interactive sliders** – Adjust G1, G2, study time, absences, and past failures in real time
- 📊 **Live prediction** – Predicted final grade with letter grade badge (A–F) and progress bar
- 📈 **Grade distribution chart** – See where your student falls in the dataset histogram
- 🔥 **Correlation heatmap** – Pearson correlations across all key features
- 🔭 **Scatter explorer** – Pick any two features and visualize them with G3 color coding
- ✅ **Actual vs Predicted** – Residual plot and scatter for the test split
- 🏆 **Model metrics** – R², MAE, RMSE displayed live
- 📉 **Feature contribution bar chart** – Understand what drives the prediction

## Project Structure

```
student_grade_predictor/
├── app.py               # Main Streamlit application
├── student-mat.csv      # UCI dataset (Mathematics, 395 students)
├── requirements.txt     # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Clone / download this folder
# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

## Dataset

**UCI Machine Learning Repository – Student Performance**  
395 Portuguese secondary-school students studying Mathematics.  
Features used: G1, G2, studytime, failures, absences → predicts G3.

## Model

- Algorithm: **Linear Regression** (scikit-learn)
- Best model selected from 50 random train/test splits (90/10)
- Typical R² on test set: **~0.85–0.95**

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Web UI & sliders |
| scikit-learn | Linear Regression model |
| pandas / numpy | Data processing |
| matplotlib / seaborn | Visualizations |
