# facemask-detection-v1

## About

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. It was used a dataset that contains 3829 images belonging to the 2 classes.
The classes are:

- With mask;
- Without mask;

**Detect faces and determine whether they are wearing mask.**

The face mask detection models was made with two mainstream deep learning frameworks:
  1. Tensorflow 2.0
  2. Keras
 
It was used OpenCV and NumPy for image processing.

## Dataset
The following dataset was used for training and testing:
  - [Dataset](https://www.kaggle.com/thiagolermen/dataset-face-mask-detection)

The dataset was divided into a training set and a cross validation set as follows:
  - Training set: 80%
  - Cross validation set: 20%


### Data preprocessing
  1. Resizing and rescaling: all the images were resized to 224x224 and 1/255 rescaling;
  2. Data augmentation: to train the model, data augmentation was used to expand the size of the training set

## Model structure
The model was trained and tested using Google Collaboratory.
The input size of the model is 224x224 (MobileNet default input).
It was MobileNet as a pretrained model and was added two more layers for fine tuning (flatten, and dense with binary output using sigmoid activation function).

## Training and validation data
![download](https://user-images.githubusercontent.com/49798588/120581627-7f9dee00-c401-11eb-8393-d6c0ebfb9bc0.png)

