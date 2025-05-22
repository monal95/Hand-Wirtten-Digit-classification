# Hand-Wirtten-Digit-classification
**Description:**<br><br>
This is a Python-based Handwritten Digit Recognition app that allows users to draw digits on a canvas and have them recognized in real-time using a Random Forest classifier. It features a clean and interactive Tkinter GUI, supports both training and loading of a model, and utilizes the MNIST or digits dataset from scikit-learn.

**Project** **Overview:**<br><br>
This project is a Handwritten Digit Recognition System built with Python and a Tkinter GUI. It enables users to draw digits (0–9) on a canvas, which are then recognized and classified using a Random Forest machine learning model trained on the MNIST or Digits dataset.<br><br>
The core idea is to provide an interactive and user-friendly interface for real-time digit classification while also demonstrating the end-to-end application of a machine learning pipeline — including data loading, model training, evaluation, prediction, and model persistence.

**Features:**<br><br>
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
Built with Tkinter, scikit-learn, NumPy, and Pillow — no heavy frameworks.


