import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
from PIL import Image
import itertools
import time

Lenet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])

class LeNet(nn.Module):
    def __init__(self, num_class, loss_criterion, pretrained_path=None, batch_size=512, num_workers=16):
        super(LeNet, self).__init__()
        # encoder
        self.f1 = [
            nn.Conv2d(1,  6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        ]
        self.f2 = [
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84)
        ]
        self.f3 = [
            nn.ReLU(),
            nn.Linear(84, num_class)
        ]
        self.f1 = nn.Sequential(*self.f1)
        self.f2 = nn.Sequential(*self.f2)
        self.f3 = nn.Sequential(*self.f3)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.loss_criterion = loss_criterion
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.num_workers = num_workers
    def forward(self, x):
        x = self.f1(x)
        x = x.view(-1, 16 * 4 * 4)
        feature = self.f2(x)
        out = self.f3(feature)
        return out, F.normalize(feature, dim=-1)

    # train or test for several epochs
    def train_val(self, epochs, is_train, data_loader=None, X=None, y=None):
        if data_loader is None:
            dataset = TorchDataset(X, y, Lenet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if is_train:
            self.train() # for BN
        else:
            self.eval()
            epochs = 1

        for epoch in range(1, epochs+1):
            total_loss, total_correct_1, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
            with (torch.enable_grad() if is_train else torch.no_grad()):
                for data, target in data_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    out, feature = self(data)
                    loss = self.loss_criterion(out, target)

                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    total_num += data.size(0)
                    total_loss += loss.item() * data.size(0)
                    prediction = torch.argsort(out, dim=-1, descending=True)
                    total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                            .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, total_correct_1 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100

    def predict(self, data_loader=None, X=None, is_train=False):
        if data_loader is None:
            y = np.zeros(X.shape[0], dtype='long')
            dataset = TorchDataset(X, y, Lenet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.eval()
        results_out, results_feature = [], []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out, feature = self(data)
                results_out.append(out), results_feature.append(feature)
        return torch.softmax(torch.cat(results_out, axis=0), axis=1), torch.cat(results_feature, axis=0)
    
    def predict_classes(self, data_loader=None, X=None, is_train=False):
        return self.predict(data_loader, X, is_train)[0].argmax(axis=1)

class TorchDataset(Dataset):
    def __init__(self, images_np, label, transform=None, dataset=None):
        self.images_np = images_np
        self.dataset = dataset
        self.label = torch.LongTensor(label) #.flatten()
        self.transform = transform

    def __len__(self):
        if self.images_np is not None:
            return len(self.images_np)
        elif self.dataset is not None:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.images_np is not None:
            img, label = self.images_np[idx], self.label[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        elif self.dataset is not None:
            img, label = self.dataset[idx][0], self.label[idx] # Ignore transform
            return img, label


if __name__ == '__main__':
    batch_size, epochs = 128, 50

    train_data = MNIST(root='data', train=True, download=True,transform=Lenet_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_data = MNIST(root='data', train=False, download=True,transform=Lenet_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = LeNet(num_class=10, loss_criterion=nn.CrossEntropyLoss(), batch_size=batch_size).cuda()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1 = model.train_val(4, True, data_loader=train_loader)
        test_loss, test_acc_1 = model.train_val(1, False, data_loader=test_loader)
        # train_loss, train_acc_1 = model.train_val(3, True, X=X_train, y=y_train)
        # test_loss, test_acc_1 = model.train_val(1, False, X=X_test, y=y_test)
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/linear_model.pth')
        #y_pred, feature = model.predict(X=X_test)
