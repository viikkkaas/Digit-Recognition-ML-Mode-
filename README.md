Handwritten Digit Recognition using CNN (MNIST Dataset)


📌 Overview

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from the MNIST dataset.
The model achieves ~98% accuracy and demonstrates the use of deep learning techniques for image classification tasks.
An additional OpenCV-based interface allows users to draw digits and test predictions in real time.


🚀 Features

Trains a CNN on the MNIST dataset for digit recognition.
Achieves high validation accuracy (~98%).
Includes OpenCV interface for user-drawn digit input.
Built using TensorFlow and Keras.
Lightweight and easy to replicate.


🧩 Technologies Used

Python
TensorFlow / Keras
NumPy, Matplotlib, OpenCV
Jupyter Notebook


📂 Project Structure

Digit-Recognition/
│
├── dataset/                 # MNIST dataset (auto-downloaded)
├── model/                   # Saved model files
├── notebooks/
│   └── digit_recognition_cnn.ipynb   # Training notebook
├── main.py                  # Main script for testing user input
├── requirements.txt         # Required dependencies
└── README.md                # Project documentation


🧠 Model Architecture

Input Layer: 28x28 grayscale images
Convolutional Layers: Two layers with ReLU activation
Pooling Layer: Max pooling for feature reduction
Fully Connected Layers: Dense layers with dropout for regularization
Output Layer: Softmax activation (10 classes for digits 0–9)


📊 Training Details

Dataset: MNIST (60,000 training + 10,000 testing images)
Optimizer: Adam
Loss Function: Categorical Crossentropy
Epochs: 50 (tunable)
Batch Size: 64



💻 How to Run

Clone the repository

git clone https://github.com/viikkkaas/digit-recognition.git
cd digit-recognition


Install dependencies

pip install -r requirements.txt


Train the model

python train_model.py


Run the real-time digit recognition interface

python main.py



🧪 Results

Training Accuracy: ~99%
Validation Accuracy: ~98%
Tested on: User-drawn digits via OpenCV interface


📈 Future Improvements

Deploy model using Flask / FastAPI for web-based predictions.
Integrate with Streamlit for interactive UI.
Experiment with custom handwritten datasets for better generalization.


👨‍💻 Author

Vikas Patil
mail: vikas.p.2706@gmail.com
