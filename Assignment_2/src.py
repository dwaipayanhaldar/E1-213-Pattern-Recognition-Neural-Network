import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
torch.manual_seed(25)
torch.mps.manual_seed(25)
from torchvision.io import read_image
device = "mps" if torch.mps.is_available() else "cpu"


#################################################
# Basic Utilities #
#################################################

def is_it_nan(val):
    return val != val
    
def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, max_iter=100, patience=10, backward_hook = False):
    """
    Basic training Loop, Applicable for All the problems.
    Early stopping: halts if validation loss does not improve for `patience` consecutive epochs.
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    grad_norms = {'layer1': [], 'layer2': [], 'layer3': []}
    hooks = []
    if backward_hook:
        max_iter = 1
        linear_layers = [m for m in model.linear_relu_stack if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            name = f'layer{i+1}'
            def make_hook(n):
                def hook(grad):
                    grad_norms[n].append(grad.norm().item())
                return hook
            hooks.append(layer.weight.register_hook(make_hook(name)))

    for iter in range(max_iter):
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.float().to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss

        train_loss /= len(train_dataloader)

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.float().to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= len(val_dataloader)

        if iter % 10 == 0:
            print(f"Epoch {iter}: Training Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}")

        # Early stopping check
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {iter}. Best Validation Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state)
                for h in hooks:
                    h.remove()
                return grad_norms if backward_hook else None

    for h in hooks:
        h.remove()
    return grad_norms if backward_hook else None


def test_loop(dataloader, model, loss_fn, classify=False, Logit_loss=False):
    """
    Basic Test Loop applicable for all the cases.
    Returns (prob_list, y_list) when classify=True, for PR curve plotting.
    """
    model.eval()
    test_loss = 0
    prediction_list = []
    prob_list = []      
    y_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if classify:
                if Logit_loss:
                    probs = torch.sigmoid(pred)
                    prediction_list.extend((pred > 0).int().tolist())
                else:
                    probs = pred
                    prediction_list.extend((pred > 0.5).int().tolist())
                prob_list.extend(probs.tolist())
                y_list.extend(y.int().tolist())

    if classify:
        print(confusion_matrix(y_list, prediction_list))
        return prob_list, y_list
    else:
        test_loss /= len(dataloader)
        print(f"Test Loss (MSE): {test_loss:.4f}")
    




#################################################
# Phase 1 #
#################################################

class TimeSeriesDataset(Dataset):
    """
    The Dataset definition for time series data specifically delhi_aqi.csv
    """
    
    def __init__(self, file_path, seq_len, hop, col_name='pm2_5', classify=False, all_features=False):
        df = pd.read_csv(file_path)

        if all_features:
            feature_cols = [c for c in df.columns if c != 'date']
            data = torch.tensor(df[feature_cols].values, dtype=torch.float32)  # (T, F)
            pm2_5_idx = feature_cols.index(col_name)
            self.input_dim = seq_len * len(feature_cols)
            self.num_features = len(feature_cols)
        else:
            data = torch.tensor(df[col_name].values, dtype=torch.float32)      # (T,)
            pm2_5_idx = None
            self.input_dim = seq_len
            self.num_features = 1

        self.X = []
        self.y = []

        for i in range(len(data) - seq_len - hop):
            x_seq = data[i:i+seq_len]
            y_target = data[i+seq_len+hop, pm2_5_idx] if all_features else data[i+seq_len+hop]

            self.X.append(x_seq)  # shape: (seq_len, F) or (seq_len,) — do NOT flatten; MLP flattens itself                
            if classify:
                self.y.append(y_target > 200)
            else:
                self.y.append(y_target)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Temporal_MLP(nn.Module):

    def __init__(self, input_dim=72, sigmoid=False, classify=False):
        super().__init__()
        self.flatten = nn.Flatten()
        if sigmoid:
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            )
        elif classify:
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
            )
        else:
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x).squeeze(1)
    

#################################################
# Phase 2 #
#################################################

class PlantVillageSeverityDataset(Dataset):
    """
    Loads PlantVillage images and synthesizes a continuous severity label:
        severity = bounding_box_area_of_necrotic_pixels / total_leaf_area
    Necrotic pixels = non-background pixels where green channel does NOT dominate.
    Returns (image_tensor, severity_float).
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        # Walk all class subdirectories
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))

        # Pre-compute all severity labels once to avoid reloading every epoch
        print(f"Pre-computing severity for {len(self.image_paths)} images...")
        resize = transforms.Resize((128, 128))
        self.severities = []
        for i, path in enumerate(self.image_paths):
            img_tensor = read_image(path)[:3].float() / 255.0  # (C, H, W) in [0,1]
            if img_tensor.shape[0] == 1:                        # grayscale → RGB
                img_tensor = img_tensor.repeat(3, 1, 1)
            img_tensor = resize(img_tensor)
            self.severities.append(self._compute_severity(img_tensor))
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(self.image_paths)}")
        print("Done.")

    def _compute_severity(self, img_tensor):
        # img_tensor: (3, H, W) float in [0, 1]
        R, G, B = img_tensor[0], img_tensor[1], img_tensor[2]

        # Leaf mask: exclude near-black background
        leaf_mask = img_tensor.sum(dim=0) > 0.15

        leaf_area = leaf_mask.sum().float()
        if leaf_area == 0:
            return torch.tensor(0.0)

        # Necrotic: leaf pixels where green is NOT the dominant channel
        green_dominant = (G > R) & (G > B)
        necrotic_mask = leaf_mask & ~green_dominant

        # Bounding box of necrotic region
        rows = necrotic_mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = necrotic_mask.any(dim=0).nonzero(as_tuple=True)[0]

        if rows.numel() == 0 or cols.numel() == 0:
            return torch.tensor(0.0)

        bbox_area = ((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)).float()
        severity = (bbox_area / leaf_area).clamp(0.0, 1.0)
        return severity

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0  # (C, H, W) in [0,1]
        if img_tensor.shape[0] == 1:                                          # grayscale → RGB
            img_tensor = img_tensor.repeat(3, 1, 1)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, self.severities[idx]

class ownCNN(nn.Module):

    def __init__(self, input_dim, output_classes):
        super().__init__()
        sp = input_dim // 32  
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  nn.BatchNorm2d(16) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.fc = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(256 * sp * sp, 256),nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, output_classes),
        ])


    def forward(self, x, print_shapes=False):
        if print_shapes:
            print(f"Input:          {tuple(x.shape)}")
        for layer in self.conv_layers:
            x = layer(x)
            if print_shapes and not isinstance(layer, nn.ReLU):
                print(f"{layer.__class__.__name__:12s}: {tuple(x.shape)}")
        for layer in self.fc:
            x = layer(x)
            if print_shapes and not isinstance(layer, nn.ReLU):
                print(f"{layer.__class__.__name__:12s}: {tuple(x.shape)}")
        if print_shapes:
            print(f"Output:         {tuple(x.shape)}")
        return x.squeeze(1) 

    def receptive_field(self):
        r_l = 1 
        j_l = 1
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                r_l += 2*j_l
                j_l *= 1 
            elif isinstance(layer, nn.MaxPool2d):
                r_l += 1*j_l
                j_l *= 2

        return r_l

#################################################
# Phase 3 #
#################################################

def train_loop_rnn(train_dataloader, model, loss_fn, optimizer):
    """
    Basic training Loop. Mainly for 3.9
    """
    loss = torch.tensor(0.0)                                   
    prev_state = {k: v.clone() for k, v in model.state_dict().items()}  
    i = 0

    while not is_it_nan(loss.item()):
        prev_state = {k: v.clone() for k, v in model.state_dict().items()}  
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.float().to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss

        train_loss /= len(train_dataloader)
        i += 1
        print(f"Epoch Number:{i}")

    model.load_state_dict(prev_state)
    svd = torch.linalg.svd(model.linearh.weight.detach().cpu())
    return svd.S[0].item()

class OwnVanillaRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.linearx = nn.Linear(input_dim, hidden_dim)
        self.linearh = nn.Linear(hidden_dim, hidden_dim)
        self.lineary = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsequeeze(0)
            x = x.unsqueeze(-1)
        batch_size, seq_length, _ = x.shape
        
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.linearx(x_t) + self.linearh(h_t))
        
        y = self.lineary(h_t)
        return y.squeeze(1)
    
    def bptt_decay(self, seq_len=100, input_dim=1, hidden_dim=64):
        x = torch.randn(1, seq_len, input_dim, device=device) 

        h_t = torch.zeros(1, hidden_dim, device=device)
        hidden_states = {}
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.linearx(x_t) + self.linearh(h_t))
            h_t.retain_grad()      
            hidden_states[t] = h_t    
        

        y = self.lineary(h_t)
        target = torch.zeros(1, device=device)
        loss = nn.MSELoss()(y.squeeze(1), target)
        
        loss.backward()
        grad_t100 = hidden_states[99].grad.norm().item()
        grad_t50  = hidden_states[49].grad.norm().item()
        grad_t0   = hidden_states[0].grad.norm().item()
        
        print(f"||dL/dh|| at t=100: {grad_t100:.6f}")
        print(f"||dL/dh|| at t=50:  {grad_t50:.6f}")
        print(f"||dL/dh|| at t=0:   {grad_t0:.6f}")
        
        return hidden_states



#################################################
# Phase 4 #
#################################################

class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        


if __name__ == "__main__":
    DATA_DIR = '/Users/dwaipayanhaldar/Downloads/Notes and Books/IISc Coding Assignments and Project/Data/PRNN_2026_A2_data'

    image_folder_path = os.path.join(DATA_DIR, 'plantvillage_dataset/segmented')

    temporal_data_folder_path = os.path.join(DATA_DIR, 'delhi_aqi.csv')
    # transform = transforms.Resize((128, 128))
    
    # image_dataset = PlantVillageSeverityDataset(image_folder_path, transform=transform)

    # image, label = image_dataset[0]
    
    # model_cnn = ownCNN(128, 38)

    # pred = model_cnn(image.unsqueeze(0), print_shapes=True)

    # print(model_cnn.receptive_field())

    temporal_dataset = TimeSeriesDataset(temporal_data_folder_path, seq_len=72, hop=24)
    N = len(temporal_dataset)
    # index = np.random.permutation(N)
    
    train_indices = list(range(int(0.7*N)))
    val_indices   = list(range(int(0.7*N), int(0.85*N)))
    test_indices  = list(range(int(0.85*N), N))

    train_dataset = Subset(temporal_dataset, train_indices)
    val_dataset   = Subset(temporal_dataset, val_indices)
    test_dataset  = Subset(temporal_dataset, test_indices)


    train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size= 32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size= 32, shuffle=False)

    rnn_model = OwnVanillaRNN(1, 64, 1).to(device)
    learning_rate = 100
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(rnn_model.parameters(), lr=learning_rate)
    epochs = 100
    print("Entering training loop")
    singular_value = train_loop_rnn(train_dataloader, rnn_model,loss_fn,optimizer)
    print(f"The largest singular value is {singular_value}")


