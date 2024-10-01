# Image Classification with Keras

This repository contains a simple convolutional neural network (CNN) model built using TensorFlow and Keras to classify fresh and stale images of fruits and vegetables. The project consists of two main scripts:

- `Classifier.py`: Builds, trains, and saves the model.
- `Usage.py`: Loads the saved model to make predictions on new images.

The dataset used for training can be found on Kaggle: [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables).

## Requirements

Ensure you have the following Python libraries installed:

- numpy
- pandas
- tensorflow
- keras
- opencv-python
- pillow

You can install the required libraries using the following command:

```bash
pip install numpy pandas tensorflow keras opencv-python
```

## Dataset

Download the dataset from Kaggle:  
[Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)

Once downloaded, place the dataset in a directory called `Resources` so the script can access it.

## Classifier.py

The `Classifier.py` script performs the following tasks:

- Loads and augments the dataset using `ImageDataGenerator`.
- Defines a CNN model with convolutional, max pooling, and dense layers.
- Trains the model on the dataset.
- Saves the trained model as `classifier_model2.keras`.

### Usage

To train the model, simply run:

```bash
python Classifier.py
```

The model will be trained for 5 epochs, and a summary of the model architecture will be displayed. After training, the model will be saved as `classifier_model2.keras`.

## Usage.py

The `Usage.py` script loads the saved model and predicts the class of a new image.

### Usage

To predict the class of an image, run the script:

```bash
python Usage.py
```

Ensure to specify the correct path for the image you want to predict. By default, the script is set to predict the class of the image located at `Resources/fresh_tomato/DSCN4068.jpg_0_1176.jpg`.

The model will print the predicted label along with the actual label.

## Model Architecture

The CNN model in `Classifier.py` consists of the following layers:

- Conv2D and MaxPooling layers for feature extraction
- Flatten and Dense layers for classification
- Softmax activation in the output layer for multi-class classification (6 categories)

