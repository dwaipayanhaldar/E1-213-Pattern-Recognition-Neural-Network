import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
import torchvision.transforms.functional as TF
torch.manual_seed(25)
torch.mps.manual_seed(25)
from torchvision.io import read_image
device = "mps" if torch.mps.is_available() else "cpu"


#################################################
# Basic Utilities #
#################################################

def is_it_nan(val):
    """Utility to check if a value is NaN, since NaN != NaN."""
    return val != val

def trainable_parameters(model):
    """Specifically written for RNN and LSTM as it asks to use state_dict() instead of parameters()"""
    number_of_params = 0
    for param_tensor, value in model.state_dict().items():
        number_of_params += value.numel()
    return number_of_params

def count_trainable(model):
    """General utility to count trainable parameters in any model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total

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
    pred_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_list.extend(pred.tolist())
            if classify:
                if Logit_loss:
                    probs = torch.sigmoid(pred)
                    prediction_list.extend((pred > 0).int().tolist())
                else:
                    probs = pred
                    prediction_list.extend((pred > 0.5).int().tolist())
                prob_list.extend(probs.tolist())
                y_list.extend(y.int().tolist())
            else:
                y_list.extend(y.tolist())

    if classify:
        print(confusion_matrix(y_list, prediction_list))
        return prob_list, y_list
    else:
        test_loss /= len(dataloader)
        print(f"Test Loss {loss_fn.__class__.__name__}: {test_loss:.4f}")
        return pred_list, y_list
    




#################################################
# Phase 1 #
#################################################

class TimeSeriesDataset(Dataset):
    """
    The Dataset definition for time series data specifically delhi_aqi.csv
    """
    
    def __init__(self, file_path, seq_len, hop, col_name='pm2_5', classify=False, all_features=False, train_frac=0.7):
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

        # Normalise using training portion stats only — prevents leakage into val/test
        n_train = int(train_frac * len(data))
        self.X_mean = data[:n_train].mean(dim=0)
        self.X_std  = data[:n_train].std(dim=0).clamp(min=1e-8)
        data = (data - self.X_mean) / self.X_std

        # PM2.5 stats for denormalising predictions
        self.y_mean = self.X_mean[pm2_5_idx] if all_features else self.X_mean
        self.y_std  = self.X_std[pm2_5_idx]  if all_features else self.X_std

        self.X = []
        self.y = []

        for i in range(len(data) - seq_len - hop):
            x_seq = data[i:i+seq_len]
            y_target = data[i+seq_len+hop, pm2_5_idx] if all_features else data[i+seq_len+hop]

            self.X.append(x_seq)  # shape: (seq_len, F) or (seq_len,) — do NOT flatten; MLP flattens itself
            if classify:
                self.y.append(y_target > (200 - self.y_mean) / self.y_std)  # 200 threshold in normalised space
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

class PlantVillageDataset(Dataset):
    """
    Loads PlantVillage images and gives class labels.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.y = []

        class_names = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.y.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return self.transform(img_tensor) if self.transform else img_tensor, label

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
        sp = input_dim // (2**3)  
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  nn.BatchNorm2d(8) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16) ,nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32) ,nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.fc = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(32 * sp * sp, 256),nn.ReLU(),
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
    
    def bptt_decay(self, seq_len=100, input_dim=1):
        # MPS does not support retain_grad() on intermediate tensors — run on CPU
        cpu = torch.device('cpu')
        self.to(cpu)

        hidden_dim = self.hidden_dim
        x = torch.randn(1, seq_len, input_dim, device=cpu)

        h_t = torch.zeros(1, hidden_dim, device=cpu)
        hidden_states = {}

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.linearx(x_t) + self.linearh(h_t))
            h_t.retain_grad()
            hidden_states[t] = h_t

        y = self.lineary(h_t)
        target = torch.zeros(1, device=cpu)
        loss = nn.MSELoss()(y.squeeze(1), target)

        loss.backward()
        grad_t100 = hidden_states[99].grad.norm().item()
        grad_t50  = hidden_states[49].grad.norm().item()
        grad_t0   = hidden_states[0].grad.norm().item()

        print(f"||dL/dh|| at t=100: {grad_t100:.2e}")
        print(f"||dL/dh|| at t=50:  {grad_t50:.2e}")
        print(f"||dL/dh|| at t=0:   {grad_t0:.2e}")

        self.to(device)
        return hidden_states



#################################################
# Phase 4 #
#################################################

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)

    def bptt_decay(self, seq_len=100, input_dim=1):
        cpu = torch.device('cpu')
        self.to(cpu)

        hidden_dim = self.lstm.hidden_size
        x = torch.randn(1, seq_len, input_dim, device=cpu)

        h_t = torch.zeros(1, 1, hidden_dim, device=cpu)
        c_t = torch.zeros(1, 1, hidden_dim, device=cpu)
        hidden_states = {}

        for t in range(seq_len):
            x_t = x[:, t:t+1, :]                      
            _, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))
            h_t.retain_grad()
            hidden_states[t] = h_t

        y = self.fc(h_t.squeeze(0))                      
        target = torch.zeros(1, 1, device=cpu)
        loss = nn.MSELoss()(y, target)

        loss.backward()
        grad_t100 = hidden_states[99].grad.norm().item()
        grad_t50  = hidden_states[49].grad.norm().item()
        grad_t0   = hidden_states[0].grad.norm().item()

        print(f"||dL/dh|| at t=100: {grad_t100:.2e}")
        print(f"||dL/dh|| at t=50:  {grad_t50:.2e}")
        print(f"||dL/dh|| at t=0:   {grad_t0:.2e}")

        self.to(device)   # restore model to original device after analysis
        return hidden_states


class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)

class EncoderDecoderLSTM(nn.Module):
    """An Encoder-Decoder architecture using LSTM for multi-step forecasting. The encoder processes the input sequence and produces a context vector (final hidden state), which is then used by the decoder to generate the output sequence one step at a time."""

    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_len):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn. LSTMCell(output_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_seq_len = output_seq_len
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)         
        _, (h_n, c_n) = self.encoder(x)

        h = h_n.squeeze(0)                   
        c = c_n.squeeze(0)                 

        decoder_input = x[:, -1, :]         

        outputs = []
        for _ in range(self.output_seq_len):
            h, c = self.decoder(decoder_input, (h, c)) 
            pred = self.fc(h)                          
            outputs.append(pred)
            decoder_input = pred                         

        return torch.stack(outputs, dim=1).squeeze(-1)  

class EncoderDecoderDataset(Dataset):
    """
    The Dataset definition for time series data specifically delhi_aqi.csv for Encoder-Decoder models, where each sample consists of an input sequence and a target sequence.
    """
    
    def __init__(self, file_path, seq_len, output_seq_len, col_name='pm2_5', train_frac=0.7):
        df = pd.read_csv(file_path)

        data = torch.tensor(df[col_name].values, dtype=torch.float32)
        self.input_dim = seq_len
        self.num_features = 1

        # Normalise using training portion stats only — prevents leakage into val/test
        n_train = int(train_frac * len(data))
        self.y_mean = data[:n_train].mean()
        self.y_std  = data[:n_train].std().clamp(min=1e-8)
        data = (data - self.y_mean) / self.y_std

        self.X = []
        self.y = []

        for i in range(len(data) - seq_len - output_seq_len):
            x_seq = data[i:i+seq_len]
            y_target = data[i+seq_len:i+seq_len+output_seq_len]
            self.X.append(x_seq)
            self.y.append(y_target)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#################################################
# Phase 5 #
#################################################


class SingleHeadTansformer(nn.Module):

    def __init__(self, input_dim, model_dim, output_dim):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.project_to_Q = nn.Linear(model_dim, model_dim)
        self.project_to_K = nn.Linear(model_dim, model_dim)
        self.project_to_V = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)   
        x = self.input_projection(x)
        Q = self.project_to_Q(x)
        K = self.project_to_K(x)
        V = self.project_to_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        output = self.output_projection(attended_values[:, -1, :])
        return output.squeeze(1)
    
    def get_attention_weights(self, x):
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = self.input_projection(x)
            Q = self.project_to_Q(x)
            K = self.project_to_K(x)

            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            weights = torch.softmax(attention_scores, dim=-1)
        return weights.cpu().numpy()
    
class SingleHeadTransformerwithSinusoidalEmbeddings(nn.Module):

    def __init__(self, input_dim, model_dim, output_dim, max_len=72):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.register_buffer('sinusoidal_embedding', self._create_sinusoidal_embeddings(model_dim, max_len))
        self.project_to_Q = nn.Linear(model_dim, model_dim)
        self.project_to_K = nn.Linear(model_dim, model_dim)
        self.project_to_V = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def _create_sinusoidal_embeddings(self, model_dim, max_len):
        pe = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()          
        div = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)                                        

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x + self.sinusoidal_embedding[:, :x.size(1), :]
        Q = self.project_to_Q(x)
        K = self.project_to_K(x)
        V = self.project_to_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        output = self.output_projection(attended_values[:, -1, :])
        return output.squeeze(1)

    def get_attention_weights(self, x):
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = self.input_projection(x)
            x = x + self.sinusoidal_embedding[:, :x.size(1), :] 
            Q = self.project_to_Q(x)
            K = self.project_to_K(x)

            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            weights = torch.softmax(attention_scores, dim=-1)
        return weights.cpu().numpy()
    

class PlantVillageDatasetViT(Dataset):
    """
    Loads PlantVillage images and gives class labels.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.y = []
        self.X = []

        class_names = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.y.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        patches = []
        for i in range(0, img_tensor.shape[1], 16):
            for j in range(0, img_tensor.shape[2], 16):
                patch = img_tensor[:, i:i+16, j:j+16]
                if patch.shape[1] == 16 and patch.shape[2] == 16:
                    patches.append(patch.flatten())  

        patches = torch.stack(patches)               
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return patches, label


#################################################
# Phase 6 #
#################################################


class focal_loss(nn.Module):
    """Focal loss for multi-class classification."""
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)
        p_t = torch.exp(-ce_loss)
        return ((1 - p_t) ** self.gamma * ce_loss).mean()

def train_loop_val_loss(train_dataloader, val_dataloader, model, loss_fn, optimizer, max_iter=100, patience=10):
    """
    Basic training Loop, Applicable for those applications that require the history of validation loss
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    val_loss_history = []

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
        val_loss_history.append(test_loss)

        print(f"Epoch {iter}: Training Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {iter}. Best Validation Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state)

    return val_loss_history
    

class AptosDataset(Dataset):
    """
    Loads Aptos images and gives class labels.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.y = []

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            img_path = os.path.join(root_dir, row['id_code'] + '.png')
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.y.append(row['diagnosis'])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return self.transform(img_tensor) if self.transform else img_tensor, label






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

    # temporal_dataset = TimeSeriesDataset(temporal_data_folder_path, seq_len=72, hop=24)
    # N = len(temporal_dataset)
    # # index = np.random.permutation(N)
    
    # train_indices = list(range(int(0.7*N)))
    # val_indices   = list(range(int(0.7*N), int(0.85*N)))
    # test_indices  = list(range(int(0.85*N), N))

    # train_dataset = Subset(temporal_dataset, train_indices)
    # val_dataset   = Subset(temporal_dataset, val_indices)
    # test_dataset  = Subset(temporal_dataset, test_indices)


    # train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size= 32, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size= 32, shuffle=False)

    # rnn_model = OwnVanillaRNN(1, 64, 1).to(device)
    # learning_rate = 100
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(rnn_model.parameters(), lr=learning_rate)
    # epochs = 100
    # print("Entering training loop")
    # singular_value = train_loop_rnn(train_dataloader, rnn_model,loss_fn,optimizer)
    # print(f"The largest singular value is {singular_value}")

    model = LSTM(1, 64, 1).to(device)
    model.bptt_decay()

