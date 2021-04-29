# facemask-detection-v1

## About

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. Was used a dataset that contains 3829 images belonging to the 2 classes.
The classes are:

- With mask;
- Without mask;

**Detect faces and determine whether they are wearing mask.**

The face mask detection models was made with two mainstream deep learning frameworks:
  1. Tensorflow 2.0
  2. Keras
Was used OpenCV and NumPy for image processing.

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
The program was executed using Google Collaboratory.
The input size of the model is 224x224 (MobileNet default input).
Was used MobileNet pretrained model and was added two more layers (flatten, and dense with binary output).

## Training and validation data
![image](https://user-images.githubusercontent.com/49798588/116597553-1efc2c80-a8fc-11eb-9e7e-1a83487439fd.png)

