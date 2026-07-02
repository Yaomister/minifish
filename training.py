import torch
from nneu import NNEU
from data.dataset import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader



if __name__ == "__main__":
    training_epochs = 100

    model = NNEU()
    
    dataset = DataLoader(Dataset())

    for epoch in range(training_epochs):
        running_loss = 0
        for i, data in (dataset):
            inputs, labels = data
            optimizer = Adam(params=model.parameters(), lr=1e-3, momentum=0.9)
            optimizer.zero_grad()

            outputs = model(labels)

            loss = CrossEntropyLoss(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()







    


