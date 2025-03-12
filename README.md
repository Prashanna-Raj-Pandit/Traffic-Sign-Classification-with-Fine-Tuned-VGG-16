
# Traffic Sign Classification with Fine-Tuned VGG-16

This project implements a deep learning model to classify traffic signs using a fine-tuned VGG-16 convolutional neural network (CNN). Built with TensorFlow and Keras, it processes a dataset of traffic sign images, trains a model to recognize 43 distinct classes, and evaluates its performance. The code is provided as a single Python script (`traffic_sign_classifier.py`) that handles data downloading, preprocessing, training, evaluation, and visualization.

## Project Goals
- Download and preprocess a traffic sign dataset automatically.
- Fine-tune a pre-trained VGG-16 model for traffic sign classification.
- Train the model and evaluate its accuracy on validation and test datasets.
- Generate visualizations of training progress and sample predictions.

## Dataset
The dataset is a custom subset of the German Traffic Sign Recognition Benchmark (GTSRB), designed to demonstrate fine-tuning with a limited-size dataset. The full GTSRB contains over 50,000 images across 43 classes, with the training set alone having over 39,000 images. For this project:
- **Total Samples**: 40 images per class (43 classes).
- **Training Set**: 28 images per class (1204 images total).
- **Validation Set**: 12 images per class (516 images total).
- **Test Set**: Separate test images with labels in `Test.csv`.
- **Image Size**: Resized to 224×224 pixels (RGB).
- **Source**: [Dropbox Link](https://www.dropbox.com/s/41o9vh00rervwn9/dataset_traffic_signs_40_samples_per_class.zip?dl=1).

The dataset is downloaded and extracted during script execution into `dataset_traffic_signs_40_samples_per_class/`, with subfolders `Train`, `Valid`, and `Test`.

## Requirements
To run this project, install the following Python libraries:
tensorflow
numpy
matplotlib
pandas
requests

## Results
After running the script, results are printed to the console and saved as figures. Example outputs (update with your results):

Validation Accuracy: ~90% (placeholder; depends on run).
Test Accuracy: ~88% (placeholder; depends on run).

<img width="1232" alt="image" src="https://github.com/user-attachments/assets/b6f89f9f-4af8-47df-b029-e3af0a51fb52" />

<img width="1232" alt="image" src="https://github.com/user-attachments/assets/e10fd712-a788-4be1-a3f0-87c6bd378f78" />

