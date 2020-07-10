from torchvision import datasets, transforms, utils
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchsummary import summary
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(18 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 18 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_directory = 'images/train'
valid_directory = 'images/validation'

transformationsT = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformationsV = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.ImageFolder(train_directory, transform=transformationsT)
val_set = datasets.ImageFolder(valid_directory, transform=transformationsV)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

summary(model, (3, 56, 56))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

epochs = 50
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() * inputs.size(0)

        # Print the progress of our training
        counter += 1

    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item() * inputs.size(0)

            # Since our model outputs a LogSoftmax, find the real
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print the progress of our evaluation
            counter += 1

    # Get the average loss for the entire epoch
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = val_loss / len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy / len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


# %%


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transformationsV(image).float()
    image = Variable(image, requires_grad=True)
    return image.cuda()  # assumes that you're using GPU


image = image_loader("images/validation/active/20200710_003825.819551.jpg")
image = image[np.newaxis, :]
predit = model(image)
print(predit)

torch.onnx.export(model,  # model being run
                  image,  # model input (or a tuple for multiple inputs)
                  "skill.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})