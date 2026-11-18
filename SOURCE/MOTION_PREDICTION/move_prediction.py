import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class MotionPrediction(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            hidden_dim - hidden dimensions as an int
            out_dim - output dimensions as an int
        """
        super(MotionPrediction, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        #self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, latent_dim)
        self.out_layer = nn.Linear(latent_dim, 2)

        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.out_layer.weight, gain=0.01)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        activation1 = torch.tanh(self.layer1(obs))
        activation2 = torch.tanh(self.layer2(activation1))
        #activation3 = torch.tanh(self.layer3(activation2))
        activation3 = torch.tanh(self.layer3(activation2))
        output = self.out_layer(activation3)

        return output,activation3

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent
array_folder= parent_dir/ "model_arrays"
latents_folder= current_script_path/"latents"
#βγάζει error μερικές φορές για κάποιο λόγο. Αν ξανατρέξει θα λειτουργήσει
train_data=torch.tensor(np.load(array_folder/"train_array.npy"), dtype=torch.float32)
test_data=torch.tensor(np.load(array_folder/"test_array.npy"),dtype=torch.float32)

BATCH_SIZE=64

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM=49*2
HIDDEN_DIM=32
LATENT_DIM=8
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LR=1e-3
EPOCHS=5
OUTPUT_DIR = "output_movepredictor"

model = MotionPrediction(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
).to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_count = 0.0, 0
    for batch in train_loader:
        inputs = torch.flatten(batch[:, :-1], start_dim=1).to(DEVICE)
        targets = batch[:, -1].to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
        total_count += batch.size(0)
    train_loss = total_loss / total_count

    model.eval()
    val_loss, val_count = 0.0, 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = torch.flatten(batch[:, :-1], start_dim=1).to(DEVICE)
            targets = batch[:, -1].to(DEVICE)

            outputs, _ = model(inputs)
            loss = loss_fn(outputs, targets)

            val_loss += loss.item() * batch.size(0)
            val_count += batch.size(0)

    val_loss = val_loss / val_count

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

def extract_latents(model, test_loader,train_loader, device):
    model.eval()
    test_latents = []
    train_latents=[]
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            inputs = torch.flatten(batch[:, :-1], start_dim=1).to(DEVICE)
            targets = batch[:, -1].to(DEVICE)
            # Get the reconstructed sequence output (batch_size, seq_len, input_dim)
            _, latent_representation = model.forward(inputs)
            test_latents.append(latent_representation.cpu().numpy())
        for batch in train_loader:
            batch = batch.to(device)
            inputs = torch.flatten(batch[:, :-1], start_dim=1).to(DEVICE)
            targets = batch[:, -1].to(DEVICE)
            # Get the reconstructed sequence output (batch_size, seq_len, input_dim)
            _, latent_representation = model.forward(inputs)
            train_latents.append(latent_representation.cpu().numpy())

    return np.concatenate(train_latents, axis=0),np.concatenate(test_latents, axis=0)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
noisy_test_data=torch.tensor(np.load(array_folder/"noisy_test_array.npy"),dtype=torch.float32)
noisy_loader = DataLoader(noisy_test_data, batch_size=BATCH_SIZE, shuffle=False)
#train_latents,test_latents = extract_latents(model, test_loader,train_loader, DEVICE)
train_latents,test_latents = extract_latents(model, noisy_loader,train_loader, DEVICE)

np.save(latents_folder/f"test_latents_{LATENT_DIM}.npy", test_latents)
np.save(latents_folder/f"train_latents_{LATENT_DIM}.npy", train_latents)