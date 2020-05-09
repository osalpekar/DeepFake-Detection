import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image

data_path = "/home/ubuntu/prep_data/labels.json"
path_prefix = "/home/ubuntu/prep_data/"

label_dict = {"REAL": 0, "FAKE": 1}

class DeepfakeData(Dataset):
    def __init__(self, split):
        with open(data_path) as f:
            data = json.load(f)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_imgs = list(data.keys())
        all_lbls = list(data.values())
        if split == "train":
            self.images = all_imgs[:1950]
            self.labels = all_lbls[:1950]
        else:
            self.images = all_imgs[1950:]
            self.labels = all_lbls[1950:]

    def __getitem__(self, index):
        img_path = path_prefix + self.images[index]
        img = Image.open(img_path)
        img = self.preprocess(img)
        label_str = self.labels[index]
        label = label_dict[label_str]

        return img, label

    def __len__(self):
        return len(self.labels)

train_dataset = DeepfakeData("train")
test_dataset = DeepfakeData("test")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle = False)

print(len(train_loader))
print(len(test_loader))

model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096,2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            cost = loss.item()
            if i % 100 == 0:
                print('Epoch:' + str(epoch) + ", Iteration: " + str(i)
                    + ", training cost = " + str(cost))
        train_accuracy, _ , _, _ = calculate_accuracy(train_loader)
        test_accuracy, _, _, _ = calculate_accuracy(test_loader)
        print('Train accuracy: %f' % train_accuracy)
        print('Test accuracy: %f' % test_accuracy)

def calculate_accuracy(loader):
    total = 0
    correct = 0

    all_images = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

            all_images.append(images)
            all_preds.append(predicted.numpy())
            all_labels.append(labels)

    return 100 * correct / total, all_images, all_preds, all_labels

train()
train_accuracy, _ , _, _ = calculate_accuracy(train_loader)
test_accuracy, _, _, _ = calculate_accuracy(test_loader)

print('Train accuracy: %f' % train_accuracy)
print('Test accuracy: %f' % test_accuracy)
