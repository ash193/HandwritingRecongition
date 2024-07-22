# HandwritingRecongition
This project utilizes the MNIST dataset to build a handwriting recognition system for identifying numerical digits. The core of the project involves training a neural network model using TensorFlow and Keras, and then using this model to make predictions on handwritten digit images.

# Features
  # Model Training:

    - Utilized the MNIST dataset to train a neural network model for digit recognition.
    - Preprocessed the dataset by normalizing pixel values to enhance model performance.
    - Constructed a sequential model with two hidden layers, each containing 128 neurons with ReLU activation, and an output layer with 10 neurons using softmax activation for classification.
    - Achieved an accuracy of over 95% on the test dataset after training for 2 epochs.
  # Model Deployment:

    - Saved the trained model for future use and loaded it for prediction tasks.
  # Digit Prediction:

    - Implemented functionality to load and preprocess grayscale digit images.
    - Utilized the trained model to predict the digit in each image, outputting the predicted digit with a visualization of the image.
    - Included error handling to manage cases where images may not be identified.
# Usage
  # Training:

    - Train the model using the provided code (commented out in the script) to build and save the handwritten.model file.
  # Prediction:

    - Place digit images in the specified directory.
    - Run the script to read each image, predict the digit using the loaded model, and display the result.

# Dependencies
  - TensorFlow
  - Keras
  - OpenCV
  - NumPy
  - Matplotlib
