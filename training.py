import torch
import glob
from nneu import NNEU
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam, lr_scheduler

training_epochs = 100
learning_rate = 1e-3
split = 0.1

def set_up_dataset():
    """Load the dataset."""

    batches = glob.glob("./training/dataset_*.npz")
    assert batches, "no training data found"

    all_board_black_perspectives = []
    all_board_white_perspectives = []
    all_scores = []
    all_colors = []

    for path in batches:
        data = np.load(path)
        all_board_black_perspectives.append(data['black_perspective'])
        all_board_white_perspectives.append(data['white_perspective'])
        all_scores.append(data['scores'])
        all_colors.append(data['colors'])

    board_black_perspectives = np.concat(all_board_black_perspectives, axis=0)
    board_white_perspectives = np.concat(all_board_white_perspectives, axis=0)
    scores = np.concat(all_scores, axis=0)
    colors = np.concat(all_colors, axis=0)

    t_board_black_perspectives = torch.from_numpy(board_black_perspectives).long()
    t_board_white_perspectives = torch.from_numpy(board_white_perspectives).long()
    t_scores = torch.from_numpy(scores)
    t_colors = torch.from_numpy(colors)

    return TensorDataset(t_board_black_perspectives, t_board_white_perspectives, t_scores, t_colors)



if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = set_up_dataset()
    val_n = int(split * len(dataset))
    train_n = len(dataset) - val_n
    train_data, val_data = random_split(dataset=dataset, lengths=[train_n, val_n])

    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False)
    
    model = NNEU().to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs, eta_min=1e-5)
    loss_fn = MSELoss()

    best_val = float('inf')

    for epoch in range(1, training_epochs + 1):
        model.train()
        train_loss = 0
        n = 0
        for  board_black_perspective, board_white_perspective, score, color in train_loader:
            board_black_perspective, board_white_perspective, score, color = board_black_perspective.to(device), board_white_perspective.to(device), score.to(device), color.to(device)
            prediction = model(board_white_perspective, board_black_perspective, color)
            loss = loss_fn(prediction, score.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        train_loss /= n

        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for  board_black_perspective, board_white_perspective, score, color in val_loader:
                board_black_perspective, board_white_perspective, score, color = board_black_perspective.to(device), board_white_perspective.to(device), score.to(device), color.to(device)
                val_loss += loss_fn(model(board_white_perspective, board_black_perspective, color), score).item()

        scheduler.step()
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "weights/model.pt")

    print(f"\nDone. Best val_loss: {best_val:.6f}")
    torch.save(model.state_dict(), "weights/model_last.pt")
