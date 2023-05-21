import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import argparse
from torchvision import datasets, transforms
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import os

import umap
import umap.plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description="Train ProxyAnchorLoss")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--max_epochs", default=1, type=int, help="Maximum training length (epochs).")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--input_size", default=32, type=int, help="input size img.")
parser.add_argument("--name", default=" ", required=True, type=str, help="informação.")
args = parser.parse_args()

seed = "ProxyAnchorLoss_"+args.name
print('seed==>',seed)

result_model = list()
result_model.append("SEED::  "+str(seed)+ "\n")
result_model.append("============================= \n")

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {} ".format(
                    epoch, batch_idx, loss
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
    print(test_embeddings.shape)
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



path_train_xl = os.path.abspath(args.data_dir)
path_test = os.path.abspath(args.test)


dataset1 = datasets.ImageFolder(path_train_xl, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,
                                          batch_size=args.batch_size,
                                          shuffle=True)


dataset2 = datasets.ImageFolder(path_test, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size,
                                         shuffle=False)

net = models.resnet50(pretrained=True)
#Remove fully connected layer
modules = list(net.children())[:-1]
modules.append(nn.Flatten())
net = nn.Sequential(*modules)


model = net.to(device)
#optimizer = optim.Adam(model.parameters(), lr=0.001) #original 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = args.max_epochs


### pytorch-metric-learning stuff ###

embedding_size = 2048
print(len(dataset1.classes))
num_classes = len(dataset1.classes)

loss_func = losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32)

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, device, train_loader, optimizer, epoch)
    #test(dataset1, dataset2, model, accuracy_calculator)

print("Teste")
acc, test_embeddings, test_labels = test(dataset1, dataset2, model, accuracy_calculator)

#print(np.array(test_embeddings.cpu()))
#print(np.array(test_labels.cpu()))


view_tsne_u = TSNE(random_state=123).fit_transform(np.array(test_embeddings.cpu()))
plt.scatter(view_tsne_u[:,0], view_tsne_u[:,1], c= np.array(test_labels.cpu()), alpha=0.2, cmap='Set1')
plt.title(seed+'-tsne_Test', fontdict={'family': 'serif', 'color' : 'darkblue','size': 8})
plt.savefig(seed+'-tsne_Test.png', dpi=120)
plt.close()

"""
mapper = umap.UMAP(n_neighbors=num_classes, min_dist=0.3, metric='correlation').fit(test_embeddings.cpu())
embedding = mapper.embedding_
fig, ax = plt.subplots(1, figsize=(24, 20))
plt.scatter(*embedding.T, s=0.1, c=test_embeddings.cpu(), cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(num_classes)-0.5)
cbar.set_ticks(np.arange(100))
cbar.set_ticklabels(dataset2.classes)
plt.title(seed+'-umap-test.png');
plt.savefig(seed+'_umap_test.png', dpi=120)
plt.close()

"""

#result_model.append("============================= \n")
result_model.append("ACC_Test::  "+str(acc)+ "\n")

#arquivo = open(seed+".txt", "a")
#arquivo.writelines(result_model)
#arquivo.close()
