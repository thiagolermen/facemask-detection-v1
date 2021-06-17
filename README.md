# Face Mask Detection

![image](https://user-images.githubusercontent.com/49798588/120672034-52385b00-c468-11eb-8597-b1853bc6ad22.png)
Image from [LogMask](https://www.logmask.com/en)

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

![dataset](https://user-images.githubusercontent.com/49798588/122326299-73865600-cf02-11eb-8d61-e35ac06b7578.png)


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
The model was trained for 20 epochs with a batch-size of 32 features. The result of the accuracy and loss of model training is shown in the graph below.

![download](https://user-images.githubusercontent.com/49798588/120581627-7f9dee00-c401-11eb-8393-d6c0ebfb9bc0.png)

For testing and valuation of the model, 20% of the dataset was used for this purpose.
Thus, **97.63%** accuracy was achieved using the pre-trained model and fine tuning.

## Testing
The script Haar Cascade Frontalface Default ([OpenCV](https://github.com/opencv/opencv)) was used to locate the faces through bouding boxes. Then these localized faces were cut and sent to the model for individual valuation.

Shown below is an example of a test using face location.

![testing](https://user-images.githubusercontent.com/49798588/120670585-f3261680-c466-11eb-933d-aba36212f2c5.png)

## Improvements and optimizations
Due to the fact that the script used only recognizes faces in standard format and frontally, we can use other scripts in order to generalize the location of these faces in the images.


