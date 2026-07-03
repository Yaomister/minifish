import torch
import glob
from nneu import NNEU
from data.dataset import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

training_epochs = 100
learning_rate = 1e-3
split = 0.1


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batches = glob.glob("./training/dataset_*.npz")
    assert batches, "no training data found"



    model = NNEU()

    training_data = Dataset()

    training_loader = DataLoader(training_data)

    for epoch in range(1, training_epochs + 1):
        model.train()
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







    


