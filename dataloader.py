import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim


class RandRectDataset(Dataset):

    def __init__(self):
        self.directory = "rect/train"
        self.image_label_list = self.read_file()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        img_path, label_path = self.image_label_list[index][0], self.image_label_list[index][1]
        img = Image.open(img_path)
        img = self.transforms(img)
        c = np.genfromtxt(label_path)
        # cord = ((c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7]))
        return img, torch.from_numpy(c).float()

    def __len__(self):
        return len(self.image_label_list)

    def read_file(self):
        image_label_list = []
        imageDir = "{0}/images".format(self.directory)
        labelDir = "{0}/labels".format(self.directory)
        for file in os.listdir(imageDir):
            f = os.path.splitext(file)[0]
            image_label_list.append(
                (os.path.join(imageDir, "{0}.jpg".format(f)), os.path.join(labelDir, "{0}.txt".format(f))))
        return image_label_list


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channel = 3
        self.conv = nn.Conv2d(3, self.channel, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.channel * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.channel * 32 * 32)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    epoch_num = 100
    batch_size = 64
    train_data = RandRectDataset()
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)

    summary(model, (3, 64, 64))

    loss_func = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    for epoch in range(epoch_num):
        train_loss = 0
        model.train()
        for batch_image, batch_label in train_loader:
            inputs, labels = batch_image.to(device), batch_label.to(device)
            optimizer.zero_grad()  # Clear optimizers
            output = model.forward(inputs)  # Forward pass
            loss = loss_func(output, labels)  # Loss
            loss.backward()  # Calculate gradients (backpropogation)
            optimizer.step()  # Adjust parameters based on gradients
            train_loss += loss.item() * inputs.size(0)  # Add the loss to the training set's rnning loss

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    return model


model = train()

train_data = RandRectDataset()
inputs, labels = train_data.__getitem__(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs, _ = inputs.to(device), labels.to(device)  # Move to device
inputs = inputs[np.newaxis, :]
print(inputs.shape, labels.shape)
output = model.forward(inputs)  # Forward pass
print(labels)
print(output)

torch.onnx.export(model,  # model being run
                  inputs,  # model input (or a tuple for multiple inputs)
                  "rect.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})
