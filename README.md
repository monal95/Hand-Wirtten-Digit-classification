# Hand-Wirtten-Digit-classification
## Description:<br>
This is a Python-based Handwritten Digit Recognition app that allows users to draw digits on a canvas and have them recognized in real-time using a Random Forest classifier. It features a clean and interactive Tkinter GUI, supports both training and loading of a model, and utilizes the MNIST or digits dataset from scikit-learn.

## Project Overview:<br>
This project is a Handwritten Digit Recognition System built with Python and a Tkinter GUI. It enables users to draw digits (0–9) on a canvas, which are then recognized and classified using a Random Forest machine learning model trained on the MNIST or Digits dataset.<br><br>
The core idea is to provide an interactive and user-friendly interface for real-time digit classification while also demonstrating the end-to-end application of a machine learning pipeline — including data loading, model training, evaluation, prediction, and model persistence.

## Features:<br>
 **Interactive Drawing Canvas:**<br>
•Draw digits (0–9) using your mouse on a black canvas.<br>
•Smooth, real-time stroke rendering.<br><br>
**Digit Recognition:**<br>
•Predicts the drawn digit using a trained Random Forest model.<br>
•Displays prediction instantly on clicking the "Predict" button.<br><br>
**Model Training:**<br>
•Load and train a machine learning model directly from the GUI.<br>
•Uses MNIST dataset by default, or falls back to sklearn.datasets.load_digits if MNIST is unavailable.<br><br>
**Model Evaluation:**<br>
•Displays accuracy after training the model.<br>
•Provides feedback in the GUI status bar.<br><br>
**Model Saving and Loading:**<br>
•Automatically saves the trained model (digit_rf_model.pkl).  
•Load a previously saved model with a single click—no need to retrain every time.<br><br>
**Error Handling & Status Feedback:**
•Robust error handling with popup messages.  
•Real-time status updates in the GUI's bottom bar.<br><br>
**Image Preprocessing:**
•Captures the canvas as an image.  
•Resizes and normalizes it to match model input dimensions (28×28 grayscale).<br><br>
**Lightweight Dependencies:**  
Built with Tkinter, scikit-learn, NumPy, and Pillow — no heavy frameworks.<br><br>
## Setup Instructions

Follow these steps to run the project locally on your machine.

### 1.Clone the Repository

```bash
git clone https://github.com/your-monal95/Hand-Wirtten-Digit-classification.git
cd your-monal95
```

### 2.Create and Activate a Virtual Environment
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Or on macOS/Linux
source venv/bin/activate

```
### 3.Install Required Dependencies
```bash
pip install numpy scikit-learn pillow
```

### 4.Run the Application
```bash
python Model.py
```
---

##  Tech Stack Used

###  Programming Language
- **Python** — Core language used for building the application logic and GUI.

###  GUI
- **Tkinter** — Python’s standard GUI toolkit for building the desktop application interface.

###  Machine Learning
- **scikit-learn** — Used for training and running the **Random Forest Classifier** on handwritten digit datasets.
- **Datasets Used:**
  - `MNIST` via `fetch_openml`
  - Fallback: `sklearn.datasets.load_digits`

###  Image Processing
- **Pillow (PIL)** — Used for drawing canvas content, converting it into image arrays, resizing, and preprocessing for prediction.

###  Data Handling
- **NumPy** — Used for handling image data arrays and numerical operations.

###  Model Persistence
- **pickle** — Saves and loads the trained model (`digit_rf_model.pkl`) for reuse without retraining.

###  Threading
- **threading** — Used to train the model in the background without freezing the GUI.

---

##  Usage

Once the application is running, follow these steps to interact with it:

###  Drawing a Digit
- Use your mouse to draw a digit (0–9) in the black square canvas area.
- The brush size is large enough to mimic thick handwriting strokes.

###  Predicting the Digit
- After drawing, click the **Predict** button.
- The model will process your drawing and display the predicted digit in bold text.

###  Clearing the Canvas
- Click **Clear Canvas** to erase your drawing and start again.

###  Training the Model
- If there's no pre-trained model (`digit_rf_model.pkl`), click **Train Model**.
- The app will:
  - Attempt to load the MNIST dataset via `fetch_openml`.
  - Fall back to a smaller digits dataset if MNIST fails.
  - Train a `RandomForestClassifier`.
  - Save the trained model to disk.

###  Loading a Saved Model
- Click **Load Model** to load a previously saved model (`digit_rf_model.pkl`).
- If no model is found, you will be prompted to train one.

---
---

## 📸 Screenshots

### 🎨 Application Interface
![App Interface](images/app_interface.png)

### 🔢 Digit Prediction Example
![Prediction Example](images/prediction_example.png)

---




