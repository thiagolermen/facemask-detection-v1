# My libraries
from FaceMaskDataset import *
from Model import *
from FaceMaskDetection import *
# Data preprocessing
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Model
import torch
import torch.nn as nn
import torch.optim as optim


def check_gpu_availability():
    if torch.cuda.is_available():
        print('training on GPU')
        device = torch.device("cuda")
    else:
        print('training on CPU')
        device = torch.device("cpu")
    return device


def create_transformations():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # (value - mean) / std
    ])


def initialize_dataset(my_transforms):
    return FaceMaskDataset(csv_file='../../dataset/dataset.csv', root_dir='../../dataset/', transform=my_transforms)


def split_data(dataset):
    train_ds, test_ds = torch.utils.data.random_split(dataset, [int((0.8*len(dataset))), int((0.2*len(dataset))+1)])
    return train_ds, test_ds


def create_loader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def main(trainable=False):

    batch_size = 32

    # Create the transformations for data augmentation
    # rescaling and resizing
    my_transforms = create_transformations()
    print('Transforms')

    # Initialize dataset using the previous transformations,
    # split the data and train and validation set and
    # create its loaders
    dataset = initialize_dataset(my_transforms)
    train_ds, test_ds = split_data(dataset)
    train_dl = create_loader(train_ds, batch_size)
    test_dl = create_loader(test_ds, batch_size)
    print('Create dataset', len(dataset))

    # If GPU is available we train using it, if it's not, use CPU
    device = check_gpu_availability()

    # Shows the first batch of augmented images
    class_names = ['with_mask', 'without_mask']
    inputs, classes = next(iter(train_dl))
    out = torchvision.utils.make_grid(inputs)
    # title =[class_names[x] for x in classes]
    imshow(out)

    # If the model should be trained we crate a new model, and train it
    # if it's not, we just load a saved model
    if trainable:
        model = initialize_model()
        # put model to GPU (if available)
        model = model.to(device)
        # choosing optimizer and loss Function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss = nn.CrossEntropyLoss()
        # train the model first 50 epochs
        train(model, optimizer, loss, train_dl, test_dl, 50, device)
        # test the model and evaluate it
        test_model(model, test_dl, device)
        # save the model
        torch.save(model, "saved_model/firstFifty.pth")
    else:
        my_model = torch.load("saved_model/firstFifty.pth")
        # Evaluate and test the model with a given image
        test_model(my_model, train_dl, device)
        #face_detection('../../dataset/test1.jpg', my_model, my_transforms)


main()
