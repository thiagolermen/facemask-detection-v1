# Model
import torch
import torch.nn as nn
import torch.nn.functional as F
# Data visualization
from torch.utils.tensorboard import SummaryWriter
# Train and test
from tqdm.notebook import tqdm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def initialize_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

    # freezing the initial layers
    for param in model.parameters():
        param.requires_grad = False

    # fine tuning and transfer learning
    model.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1)
    )

    return model


# Train
#   * Input: model, optimizer, loss_fn, train_loader, val_loader, epochs
#   * Fit the model in the training loop, calculating the training_loss (average loss)
#   saving the tensorboard logs in the 'saves/logs' folder
#   * Evaluate the model accuracy and loss using the cross validation set
#   saving the model checkpoint when training_loss < 0.005 in the 'dataset/saves/'
#   folder
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    writer = SummaryWriter()
    for epoch in tqdm(range(epochs)):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for inputs, targets in train_loader():
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))

        # save checkpoint
        if training_loss < 0.005:
            torch.save(model, 'dataset/saves/' + str(epoch) + '.pth')

        # tensorBoard save log
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Train/Loss', training_loss, epoch)
        writer.add_scalar('Test/Loss', valid_loss, epoch)
        writer.add_scalar('Test/Accuracy', num_correct / num_examples, epoch)
        writer.flush()


# Test
#   * Input: model, test_dl, device
#   * Evaluate the model using the Accuracy parameter
def test_model(model, test_dl, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Correct: {:d}  total: {:d}'.format(correct, total))
    print('Accuracy = {:f}'.format(correct / total))


def save_model(model, file_name):
    torch.save(model, 'dataset/saved_model/' + file_name + '.pth')
