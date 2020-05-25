import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from triplet_selectors import *
import numpy as np
import json
from PIL import Image
from resnet import resnet18

label_dict = {"REAL": 1, "FAKE": 0}

data_path = "/home/ubuntu/prep_data/labels.json"
path_prefix = "/home/ubuntu/prep_data/"
train_split = 0.9 # 90% of the dataset is for training

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

        train_examples = int(len(all_lbls) * 0.9)

        if split == "train":
            self.images = all_imgs[:train_examples]
            self.labels = all_lbls[:train_examples]
        else:
            self.images = all_imgs[train_examples:]
            self.labels = all_lbls[train_examples:]

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

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class cffn_net(nn.Module):
    def __init__(self):
        super(cffn_net, self).__init__()
        self.resnet = resnet18(pretrained = False)
        self.softmax = nn.Softmax()
        self.bn1 = nn.BatchNorm2d(num_features=512)
        self.conv1 = nn.Conv2d(512, 2, 3)
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x, l4_output = self.resnet(x)
        # might need to do extra fc or dense layer here
        sia_out = self.softmax(x)
        #print("resnet output shape: ")
        #print(l4_output.shape)
        x = self.bn1(l4_output)
        x = x * F.sigmoid(x) #swish activation
        x = self.conv1(x)
        print(x.shape)
        #x = torch.mean(x)
        #x = torch.flatten(x, 1)
        x = x.view(50*5*5, -1)
        print(x.shape)
        x = self.fc1(x)
        return x, sia_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = cffn_net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
margin = 1
triplet_loss = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

def train():
    for epoch in range(1):
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, siamese_output = model(inputs)
            if epoch <= 3:
                loss_outputs = triplet_loss(siamese_output, labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                loss.backward()
            else:
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
            optimizer.step()

            cost = loss.item()
            if i % 100 == 0:
                print('Epoch:' + str(epoch) + ", Iteration: " + str(i)
                    + ", training cost = " + str(cost))

train()
