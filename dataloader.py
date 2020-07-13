import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms


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
        return img, np.genfromtxt(label_path)

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


epoch_num = 1
batch_size = 8

train_data = RandRectDataset()
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
print(train_loader)

for epoch in range(epoch_num):
    for batch_image, batch_label in train_loader:
        image = batch_image[0, :]
        image = image.numpy()  # image=np.array(image)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
