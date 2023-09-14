import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import lightning.pytorch as pl
import time
import torchmetrics

start = time.time()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class BirdNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 40, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40 * 6 * 6, 1080)
        self.fc2 = nn.Linear(1080, 525)
        #self.fc3 = nn.Linear(720, 525)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 40 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.dropout(x)
        #x = self.fc3(x)
        #x = self.softmax(x)
        #print(x)
        return x
    
    def cross_entropy_loss(self, outputs, labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, labels)
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        predicted_labels = torch.argmax(outputs, dim=1)
        correct_predictions = (predicted_labels == labels).sum()
        accuracy = correct_predictions/len(labels)
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        predicted_labels = torch.argmax(outputs, dim=1)
        correct_predictions = (predicted_labels == labels).sum()
        accuracy = correct_predictions/len(labels)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        return loss

    """def validation_step(self, validation_batch, batch_idx):
        inputs, labels = validation_batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("val_loss", loss)
        return loss"""

class BirdsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=r'.\birds_525\train', batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
        self.transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def setup(self, stage=None):
        images = datasets.ImageFolder(self.data_dir, transform = self.transform)
        train_set, test_set = torch.utils.data.random_split(images, [0.8, 0.2])
        #train_set, valid_set = torch.utils.data.random_split(train_set, [0.8, 0.2])
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers)
        #self.valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size, num_workers=self.num_workers)
   
    def train_dataloader(self):
        return self.train_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    #def val_dataloader(self):
        #return self.valid_loader

if __name__ == '__main__':
    bird = BirdNet()
    birdsdata = BirdsDataModule()
    trainer = pl.Trainer(max_epochs=25, accelerator="cuda", devices="auto")
    trainer.fit(bird, birdsdata)
    trainer.test(bird, birdsdata)
    print("time elapsed: " + str(time.time()-start))