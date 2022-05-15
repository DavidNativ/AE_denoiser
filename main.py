import copy

CLEARML = False

if CLEARML:
    from clearml import Task, Logger
    from clearml import Dataset as cML_DS

import random
import argparse
from denoising_model import denoising_model


import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import time
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.


### in order to avoid a weird DLL error...
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train(model,device,train_loader,criterion, optimizer, epoch, std, mean):
    #model.train()
    mean_loss = 0
    for i, (targets, _) in enumerate(train_loader):
        #adding noise
        noisy_data = copy.deepcopy(targets)
        for k, (d) in enumerate(noisy_data):
            noisy_data[k] = add_noise(d, std, mean)

        optimizer.zero_grad()
        outputs = model(noisy_data.view(noisy_data.shape[0], -1))
        e1 = outputs.view(outputs.shape[0], -1)
        e2 = targets.view(targets.shape[0], -1)
        loss = criterion(e1,e2)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()

        # # show some samples
        # nb = 6
        # fig, ax = plt.subplots(nrows=nb, ncols=2, squeeze=True)#, sharex=True, sharey=True, squeeze=True)
        # for i in range(nb):
        #     #random indice
        #     r = random.randint(0, batch_size-1)
        #     img = targets[r].detach().numpy()
        #     noisy = data[r].detach().numpy()
        #     ax[i][0].imshow(img.reshape((28,28)))
        #     ax[i][1].imshow(noisy.reshape((28,28)))
        #     ax[i][0].axis('off')
        #     ax[i][1].axis('off')
        # plt.show()

        # optimizer.zero_grad()
        # outputs = model(data)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()
        #
        # mean_loss += loss.item()
        # if i % 20 == 0:
        #     print(f'Loss {loss.item()}')

    mean_loss /= len(train_loader)
    print(f"Mean Loss on epoch {epoch} : {mean_loss}")

      #     if i % 2000 == 0:
    #         print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({ 100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    #
    # mean_loss /= len(train_loader)
    # # logging current loss
    # if CLEARML:
    #     Logger.current_logger().report_scalar(
    #         'loss metrics',
    #         'training loss',
    #         iteration=epoch,
    #         value=mean_loss
    #     )
    return mean_loss

def test(model,device,test_loader, epoch, std, mean):
    #model.eval()
    mean_loss = 0
    accuracy = 0
    print(f"Testing epoch {epoch}")
    with torch.no_grad():
        for i, (targets, _) in enumerate(test_loader):
            # adding noise
            noisy_data = copy.deepcopy(targets)
            for k, (d) in enumerate(noisy_data):
                noisy_data[k] = add_noise(d, std, mean)

            noisy_data = noisy_data.view(noisy_data.shape[0],-1)
            targets = targets.view(targets.shape[0], -1)
            outputs = model(noisy_data).view(targets.shape[0], -1)
            loss = criterion(outputs, targets)
            mean_loss += loss.item()

            # show some samples
            nb = 6
            fig, ax = plt.subplots(nrows=nb, ncols=3, sharex=True, sharey=True, squeeze=True)
            for i in range(nb):
                # random indice
                r = random.randint(0, noisy_data.shape[0] - 1)
                orig = targets[r].view(28, 28).numpy()
                noisy = noisy_data[r].view(28, 28).numpy()
                decoded = outputs[r].view(28, 28).numpy()

                ax[i][0].imshow(orig.squeeze())
                ax[i][0].axis('off')
                ax[i][1].imshow(noisy.squeeze())
                ax[i][1].axis('off')
                ax[i][2].imshow(decoded.squeeze())
                ax[i][2].axis('off')
            plt.show()
            break

        mean_loss /= len(train_loader)
        print(f"Mean Loss on epoch {epoch} : {mean_loss}")

          #     if i % 2000 == 0:
        #         print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({ 100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        #
        # mean_loss /= len(train_loader)
        # # logging current loss
        # if CLEARML:
        #     Logger.current_logger().report_scalar(
        #         'loss metrics',
        #         'training loss',
        #         iteration=epoch,
        #         value=mean_loss
        #     )
        return mean_loss


def add_noise(tensor, std, mean):
        #gaussian noise
        return tensor + torch.randn(tensor.size()) * std + mean


if __name__ == "__main__":
    print("MNIST Denoiser AutoEncoder")

    #transforming the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        #,transforms.RandomApply([AddGaussianNoise(mean, std)], p=0.5)
    ])

    # Clear ML integration
    if CLEARML:
        task = Task.init(project_name='02_AE_MNIST_denoiser', task_name='01')

    ##### args
    # Setting Hyperparameters through a dict ....
    hyper_param_dict = {
        "batch_size": 128,
        "learning_rate": 0.01,
        "checkpoint": 3
    }
    if CLEARML:
        task.connect(hyper_param_dict)

    # setting another HP through arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, help='Number of training epochs', default=4)
    args = parser.parse_args()

    #retrieving args for easier code
    num_epochs = args.epochs
    batch_size = hyper_param_dict["batch_size"]
    learning_rate = hyper_param_dict["learning_rate"]
    #saving the model each ...
    checkpoint = hyper_param_dict["checkpoint"]

    std = 1.
    mean = 0.

    #### datasets & dataloaders
    #preparing the datasets
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data', train=False, download=True, transform=transform),

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset[0], batch_size=batch_size, shuffle=True, drop_last=True)

    #showing some samples
    # fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, squeeze=True)
    # for i in range(8):
    #     for j in range(8):
    #         #random indice
    #         r = random.randint(0, len(train_dataset))
    #         img = train_dataset[r]
    #         #adding noise randomly
    #         img_data = add_noise(img[0], std, mean, p).detach().numpy()
    #         ax[i][j].set_title(img[1])
    #         ax[i][j].imshow(img_data.squeeze())
    #         ax[i][j].axis('off')
    #         ax[i][j].axis('off')

    # plt.show()

    ### Training
    print("Training")
    model = denoising_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_loss_tab = []
    test_loss_tab  = []

    for epoch in range(num_epochs):
        _train_loss = train(model, device, train_loader, criterion, optimizer, epoch, std, mean)
        schedular.step()
        _test_loss = test(model, device, test_loader, epoch, std, mean)

        train_loss_tab.append(_train_loss)
        #test_loss_tab.append(_test_loss)

        # if epoch % checkpoint == 0:
        #     # Save Model
        #     d = time.strftime("%Y,%m,%d,_%H,%M,%S")
        #     t = d.split(',')
        #     today = ''.join(t)
        #     filename = f".\MODELS\Model_{today}_{epoch}_{num_epochs}.pth"
        #     torch.save(model.state_dict(), filename)

    print(train_loss_tab)


