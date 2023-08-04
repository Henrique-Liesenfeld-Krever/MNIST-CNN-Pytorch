import dataset
import cnn

import torch.nn
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader 
from copy import deepcopy
from matplotlib import pyplot as plt

def validation(modelo, dataloader):
    total=0
    correto = 0
    for i, (image,label) in enumerate(dataloader):
        y = modelo(image)
        value, pred = torch.max(y,1)
        total += y.size(0)
        correto += torch.sum(pred == label)
    return float(correto*100/total)

def train(epochs=10, learning_rate=1e-3):
    data = dataset.get_dataset()
    modelo = cnn.CNN()
    modelo = modelo.modelo
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelo.parameters(),lr=learning_rate)
    melhor_accuracy = 0
    melhor_modelo = modelo
    modelos = []
    
    for epoch in range(epochs):
        for i, (image,label) in enumerate(data['treino']):
            optimizer.zero_grad()
            prediction = modelo(image)
            loss = loss_func(prediction,label)
            loss.backward()
            optimizer.step()
        accuracy = validation(modelo,data['teste'])
        modelos.append(accuracy)
        if accuracy > melhor_accuracy:
            melhor_modelo = deepcopy(modelo)
            melhor_accuracy = accuracy
        print("Epoch:",epoch, "accuracy: ",accuracy)
    plt.plot(modelos)
    return melhor_modelo
        
        
if __name__ =='__main__':
    modelo_treinado = train()
            
    