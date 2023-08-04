import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataset():
    return dataloader
transform = transforms.ToTensor()
treino = torchvision.datasets.MNIST(root='dataset/',train = True, transform=transform, download = True)
teste = torchvision.datasets.MNIST(root='dataset/',train = False, transform=transform, download = True)

dataloader = {
    'treino':DataLoader(treino, batch_size=10,shuffle=True,num_workers=1),
    'teste':DataLoader(teste,batch_size=10,shuffle=True,num_workers=1)
}