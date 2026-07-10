# Project 2 — Data Classification Using AI

**DecodeLabs Industrial Training Kit — Batch 2026**

A supervised learning project that classifies Iris flowers into one of three
species (*Setosa*, *Versicolor*, *Virginica*) using a **K-Nearest Neighbors
(KNN)** classifier, served through an interactive **Streamlit** app.

This fulfills the Project 2 brief:
- Load and understand a dataset (Iris: 150 samples, 3 classes, 4 features)
- Split data into training and testing sets
- Apply a simple classification algorithm (KNN)
- Evaluate with a confusion matrix and F1 score (not just raw accuracy)

## Project Structure

```
iris_project/
├── model.py           # Training pipeline: load data, scale, split, train, evaluate
├── app.py             # Streamlit app (UI + predictions + performance dashboard)
├── requirements.txt    # Python dependencies
├── README.md
└── iris_model.pkl     # Generated on first run (trained model + scaler)
```

> Streamlit is a single-file, Python-native UI framework, so there is no
> `templates/` or `static/` folder here (that's a Flask/Django pattern).
> All layout, styling, and the equivalent of "index" and "result" pages
> live inside `app.py` as tabs ("Predict" and "Model Performance").

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Train the model manually**
   ```bash
   python model.py
   ```
   This prints the optimal K, confusion matrix, accuracy, and F1 score, and
   saves `iris_model.pkl`. If you skip this step, `app.py` will train and
   save the model automatically on first launch.

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   Then open the URL Streamlit prints (usually `http://localhost:8501`).

## How It Works (The IPO Framework)

| Stage   | What Happens |
|---------|---------------|
| Input   | Iris dataset loaded, features scaled with `StandardScaler` (mean 0, variance 1) |
| Process | Data shuffled and split (80/20 train/test), `KNeighborsClassifier` trained |
| Output  | Confusion matrix, precision/recall/F1, and live predictions from user input |

## Using the App

- **Predict tab**: move the sliders for sepal/petal length & width, click
  **Classify Flower**, and see the predicted species with a confidence
  breakdown per class.
- **Model Performance tab**: adjust K interactively and watch the confusion
  matrix, accuracy, and F1 score update — a hands-on look at the
  "Tuning the Engine: Choosing K" concept from the training kit.

## Key Concepts Demonstrated

- Supervised learning vs. hardcoded heuristic rules
- Feature scaling and why it matters (the "Gatekeeper Rule")
- Train/test split with shuffling to avoid order bias
- The K-Nearest Neighbors proximity principle
- Why accuracy alone can mislead ("Accuracy Mirage") — confusion matrix,
  precision, recall, and F1 score as better diagnostics

## Contact

DecodeLabs · www.decodelabs.tech · decodelabs.tech@gmail.com
