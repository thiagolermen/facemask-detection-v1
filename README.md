# facemask-detection-v1

## About

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. It was ussed a dataset that contains 3797 images belonging to the 2 classes.
The classes are:

- With mask;
- Without mask;

**Detect faces and determine whether they are wearing mask.**

The face mask detection models was made with two mainstream deep learning frameworks:
  1. Tensorflow 2.0
  2. Keras
It was used OpenCV and NumPy for image processing.

Training set: 80%
Cross validation set: 20%

## Model structure
The program was executed using Google Collaboratory.
The input size of the model is 224x224 (MobileNet default input).
It was used MobileNet pretrained model and it was added two more layers (flatten, and dense with binary output).

## Training and validation data
![image](https://user-images.githubusercontent.com/49798588/116450287-7c777700-a831-11eb-9c85-845fbf63550a.png)

