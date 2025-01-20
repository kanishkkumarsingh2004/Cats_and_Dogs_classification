---

# Cats vs Dogs Classification in Jupyter Notebook

This project demonstrates the process of training a neural network to classify images as either "cats" or "dogs" using PyTorch in a Jupyter Notebook environment.

## Project Overview
The notebook guides you through the following steps:
1. **Loading and Preparing the Dataset**: Images of cats and dogs are resized, converted to grayscale, and normalized.
2. **Building the Model**: A fully connected neural network is defined for the classification task.
3. **Training the Model**: The model is trained on the images using the `train` function.
4. **Evaluating the Model**: Accuracy is calculated to measure the model's performance on the training set.
5. **Making Predictions**: The trained model can predict whether an image contains a cat or a dog.

## Requirements
- matplotlib
- PyTorch
- OpenCV
- os

## Setup

1. **Install Dependencies**:
   - Install the required Python libraries by running:
     ```bash
     pip install torch torchvision opencv-python matplotlib
     ```
   
2. **Dataset**:
   - Place the dataset in the `./train` directory, with subdirectories `cats` and `dogs` for their respective images.

## Steps in the Notebook

1. **Data Preparation**: 
   - The notebook loads the images, resizes them to 60x60 pixels, and normalizes the pixel values.
   
2. **Model Architecture**:
   - A multi-layer fully connected neural network is created with ReLU activations and a final layer for classification.
   
3. **Training**:
   - The training process is run for a specified number of epochs, and loss is tracked during each epoch.

4. **Accuracy**:
   - After training, the notebook calculates the accuracy of the model on the training data.

5. **Prediction**:
   - You can make predictions by calling the `pridect()` function on any image from the dataset.

## Running the Notebook

To run the notebook:
1. Open the `.ipynb` file in Jupyter Notebook.
2. Execute each cell in sequence.
3. Follow the outputs to observe the training process and accuracy.

## License

This project is licensed under the MIT License.

---
