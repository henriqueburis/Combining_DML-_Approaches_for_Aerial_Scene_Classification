import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import argparse

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import os

import umap
import umap.plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *


parser = argparse.ArgumentParser(description="Train TripletMargiLoss")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--max_epochs", default=200, type=int, help="Maximum training length (epochs).")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--input_size", default=32, type=int, help="input size img.")
parser.add_argument("--name", default=" ", required=True, type=str, help="informação.")
args = parser.parse_args()

seed = "TripletMargiLoss_"+args.name
print('seed==>',seed)

result_model = list()
result_model.append("SEED::  "+str(seed)+ "\n")
result_model.append("============================= \n")

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 32, 1)
        self.conv2 = nn.Conv2d(32, 64, 33, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
"""


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    ) 
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    return accuracies["precision_at_1"], test_embeddings, test_labels

device = torch.device("cuda")

transform = transforms.Compose(
    [transforms.Resize((args.input_size,args.input_size)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

#batch_size = 256

#dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
#dataset2 = datasets.MNIST(".", train=False, transform=transform)
#train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)

argumento_path_train_XL = args.data_dir
#argumento_path_train_XU = "flower102_data/XU"
argumento_path_test = args.test

path_train_xl = os.path.abspath(argumento_path_train_XL)
#path_train_xu = os.path.abspath(argumento_path_train_XU)
path_test = os.path.abspath(argumento_path_test)


dataset1 = datasets.ImageFolder(path_train_xl, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,
                                          batch_size=args.batch_size,
                                          shuffle=True)

#dataset3 = datasets.ImageFolder(path_train_xu, transform=transform)
#train_loader_xu = torch.utils.data.DataLoader(dataset3,
 #                                         batch_size=32,
  #                                        shuffle=False)

dataset2 = datasets.ImageFolder(path_test, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size,
                                         shuffle=False)

net = models.resnet50(pretrained=True)
#Remove fully connected layer
modules = list(net.children())[:-1]
modules.append(nn.Flatten())
net = nn.Sequential(*modules)


model = net.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = args.max_epochs


### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
#this is the Euclidean distance  fonte: https://kevinmusgrave.github.io/pytorch-metric-learning/distances/
#distance = distances.LpDistance()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    #test(dataset1, dataset2, model, accuracy_calculator)

print("Teste")
acc, test_embeddings, test_labels  = test(dataset1, dataset2, model, accuracy_calculator)
print(acc)
#print("XU")
#test(dataset1, dataset3, model, accuracy_calculator)

view_tsne_u = TSNE(random_state=123).fit_transform(np.array(test_embeddings.cpu()))
plt.scatter(view_tsne_u[:,0], view_tsne_u[:,1], c= np.array(test_labels.cpu()), alpha=0.2, cmap='Set1')
plt.title(seed+'-tsne_Test', fontdict={'family': 'serif', 'color' : 'darkblue','size': 8})
plt.savefig(seed+'-tsne_Test.png', dpi=120)
plt.close()

result_model.append("============================= \n")
result_model.append("ACC_Test::  "+str(acc)+ "\n")

#arquivo = open(seed+".txt", "a")
#arquivo.writelines(result_model)
#arquivo.close()
