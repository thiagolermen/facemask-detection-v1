# facemask-detection-v1

## About

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. With this dataset, it is possible to create a model to detect people wearing mask or not wearing them..
This dataset contains 3797 images belonging to the 2 classes.
The classes are:

- With mask;
- Without mask;

**Detect faces and determine whether they are wearing mask.**

The face mask detection models was made with two mainstream deep learning frameworks:
  1. Tensorflow 2.0
  2. Keras
It was used OpenCV and NumPy for image processing.

## Model structure
The program was executed using Google Collaboratory.
The input size of the model is 224x224 (MobileNet default input).
It was used MobileNet pretrained model and it was added two more layers (flatten, and dense with binary output).

## Training and validation data
![image](https://user-images.githubusercontent.com/49798588/116448137-24d80c00-a82f-11eb-8173-3e82a9cd5b1e.png)
