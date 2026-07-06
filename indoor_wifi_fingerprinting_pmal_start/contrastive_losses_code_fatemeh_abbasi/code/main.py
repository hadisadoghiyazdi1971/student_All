from torchvision.transforms import ToTensor
import torch
import random
import numpy as np
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from loss_functions.TripletSemiHardLoss import TripletSemiHardLoss
from loss_functions.SupConLoss import SupConLoss
from loss_functions.TripletLoss import TripletLoss


torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

embedding_dims = 2
batch_size = 32
epochs = 50

# بارگیری داده‌های MNIST
train_dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='data/', train=False, transform=ToTensor())

# ساخت دیتالودرها
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

model = Network(embedding_dims)
model.apply(init_weights)
model = torch.jit.script(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
#criterion = TripletSemiHardLoss(device)
#criterion = SupConLoss(); sw = 1
criterion = TripletLoss()

if __name__ ==  '__main__':
    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, (imgs, labels) in enumerate \
                (tqdm(train_loader, desc="Training", leave=False)):
            imgs = imgs.to(device)
            labels = labels.to(device)



            if sw:
                imgs = torch.cat([imgs[0], imgs[1]], dim=0)
                featurs = model(imgs)
                f1, f2 = torch.split(featurs, [batch_size, batch_size], dim=0)
                featurs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            else:
                featurs = model(imgs)

            loss = criterion(featurs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, "trained_model.pth")

    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(train_loader):
            train_results.append(model(img.to(device)).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    train_results.shape

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels==label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

    test_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            test_results.append(model(img.to(device)).cpu().numpy())
            labels.append(label)

    test_results = np.concatenate(test_results)
    labels = np.concatenate(labels)
    test_results.shape

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = test_results[labels == label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

    print('end')