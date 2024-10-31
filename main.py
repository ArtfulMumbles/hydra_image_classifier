import hydra
from omegaconf import DictConfig
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # torch device
    device =  if torch.cuda.is_available() else "cpu"
    # use config to get dataset
    transform = transforms.Compose([transforms.ToTensor()])
    if cfg.dataset.name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif cfg.dataset.name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # split dataset to train and val
    train_length = int(len(dataset) * (1 - cfg.train_val_proportion))
    val_length = len(dataset) - train_length  # To ensure the total length is exact
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    # data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    # get model
    if cfg.model.name == "simple_cnn":
        model = nn.Sequential(
            nn.Conv2d(cfg.dataset.input_size[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * cfg.dataset.input_size[1] * cfg.dataset.input_size[2], cfg.dataset.num_classes)
        )
    elif cfg.model.name == "resnet":
        model = torchvision.models.resnet18(num_classes=cfg.dataset.num_classes)
    elif cfg.model.name == "vgg":
        model = torchvision.models.vgg11(num_classes=cfg.dataset.num_classes)
    model.to(device)
    # get optimizer
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.learning_rate, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    # loss criterion
    criterion = nn.CrossEntropyLoss()
    # train loop
    for epoch in range(cfg.epochs):
        for i, (images, labels) in tqdm(enumerate(train_dataloader), desc=f'Train Epoch {epoch}'):
            images, labels = images.to(device), labels.to(device)
            # model to train
            model.train()
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # opimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # eval
        total_samples = 0
        samples_correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_dataloader):
                # model to eval
                model.eval()
                # forward pass
                outputs = model(images)
                # statistics
                loss = criterion(outputs, labels) # cross entropy loss
                _, predicted = torch.max(outputs, 1)
                samples_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        # statistics
        accuracy = samples_correct / total_samples
        # print
        print(f'For epoch {epoch}: Loss = {loss}, Accuracy = {accuracy}')
            
if __name__ == '__main__':
    main()