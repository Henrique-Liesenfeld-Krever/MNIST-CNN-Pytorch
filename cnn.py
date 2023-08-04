import dataset
import torch.nn

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.modelo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        
        
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(32*5*5,10),
            torch.nn.ReLU()
        )
        
