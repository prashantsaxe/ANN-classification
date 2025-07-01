# prashantsaxe-ann-classification

Artificial Neural Network (ANN) based classification and regression on the **Churn Modelling dataset**, including:

- Data preprocessing pipelines
- Label encoding and one-hot encoding
- ANN model building, training, and hyperparameter tuning
- Regression variant using ANN
- Streamlit app for regression demo
- Experiment tracking with TensorBoard logs

---

## 📂 Project Structure

```
prashantsaxe-ann-classification/
├── app.py                       # Streamlit app for classification
├── streamlit_regression.py      # Streamlit app for regression
├── Churn_Modelling.csv          # Dataset
├── experiments.ipynb            # Initial experiments with ANN
├── hyperparametertning_ANN.ipynb # Hyperparameter tuning
├── prediction.ipynb             # Prediction workflow
├── regression.ipynb             # Regression workflow using ANN
├── model.h5                     # Trained classification ANN model
├── regression_model.h5          # Trained regression ANN model
├── label_encoder_gender.pkl     # Label encoder for gender
├── onehot_encoder_geo.pkl       # One-hot encoder for geography
├── scaler.pkl                   # Scaler for input features
├── requirements.txt             # Project dependencies
└── logs/
    └── fit/, fit1/, ...         # TensorBoard logs of experiments
```

---

## 🚀 Features

✅ **Classification using ANN** to predict customer churn  
✅ **Regression using ANN** for continuous output variants  
✅ **Hyperparameter tuning** for ANN architecture and training parameters  
✅ **Preprocessing pipeline** with scalers and encoders saved as `.pkl`  
✅ **TensorBoard logging** for loss/accuracy visualization  
✅ **Streamlit interface** to test and visualize predictions interactively

---

## 📊 Dataset

Uses **Churn_Modelling.csv** with features including:
- Geography
- Gender
- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Target: `Exited` (1 if customer churned, 0 otherwise)

---

## 🛠️ Installation

1️⃣ Clone the repository:

```bash
git clone https://github.com/prashantsaxe/prashantsaxe-ann-classification.git
cd prashantsaxe-ann-classification
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Project

### Train or test models:
Run **notebooks** (`experiments.ipynb`, `hyperparametertning_ANN.ipynb`) in Jupyter or Colab.

### Run Streamlit App:
**For classification:**

```bash
streamlit run app.py
```

**For regression:**

```bash
streamlit run streamlit_regression.py
```

---

## 🖥️ Logs and Experiments

- **TensorBoard logs** are stored under `logs/` with structured directories for multiple experiments.
- To visualize:
```bash
tensorboard --logdir logs/
```
and open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 🧩 Models and Encoders

- `model.h5`, `regression_model.h5`: Saved ANN models for classification and regression.
- `.pkl` files: Scalers and encoders used during preprocessing to ensure consistent input pipelines during inference.

---

## 🚧 Future Improvements

- Automate hyperparameter search using Optuna or Keras Tuner.
- Integrate Docker for consistent deployment environments.
- Enable CLI interface for batch predictions.

---

## 🤝 Contributions

If you find bugs or improvements, feel free to open an issue or PR!

---

## 📜 License

This project is for learning and academic experimentation purposes.
