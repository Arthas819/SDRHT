"""
    This file is a FDRO solver. 
    This method and code follow the paper "Flow-based Distributionally Robust Optimization", https://arxiv.org/pdf/2310.19253. 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq as tdeq
from sklearn.metrics import f1_score

def _get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device

# For images, do normalization to [0, 1] and then flatten the image to a vector.
def _normalize_image_shape(image_shape):
    if len(image_shape) != 3:
        raise ValueError("image_shape must be a 3-tuple.")
    if image_shape[0] <= 4:
        channels, img_rows, img_cols = image_shape
        channels_last = False
    elif image_shape[2] <= 4:
        img_rows, img_cols, channels = image_shape
        channels_last = True
    else:
        raise ValueError("image_shape must be channels-first (C,H,W) or channels-last (H,W,C).")
    return channels, img_rows, img_cols, channels_last

def _to_image_tensor(X, device, dtype=torch.float32, image_shape=(1, 28, 28)):
    if isinstance(X, torch.Tensor):
        tensor = X.detach().to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(X, dtype=dtype, device=device)

    channels, img_rows, img_cols, channels_last = _normalize_image_shape(image_shape)
    if channels_last:
        return tensor.reshape(-1, img_rows, img_cols, channels).permute(0, 3, 1, 2).contiguous()
    return tensor.reshape(-1, channels, img_rows, img_cols)


def _to_fdro_tensor(X, device, dtype=torch.float32, image_shape=(1, 28, 28)):
    if image_shape is None:
        if isinstance(X, torch.Tensor):
            return X.detach().to(device=device, dtype=dtype).view(X.shape[0], -1)
        return torch.tensor(X, dtype=dtype, device=device).view(len(X), -1)
    return _to_image_tensor(X, device, dtype, image_shape)

class CNN(nn.Module):
    def __init__(self, img_rows=28, img_cols=28, channels=1, nb_filters=64, nb_classes=2):
        super(CNN, self).__init__()
        self.activation = nn.ELU()
        self.conv1 = nn.Conv2d(channels, nb_filters, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=6, stride=2, padding=0)
        self.conv3 = nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=5, stride=1, padding=0)
        self.fc = nn.Linear(self._calc_input_feats(img_rows, img_cols, nb_filters), nb_classes)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _calc_input_feats(self, img_rows, img_cols, nb_filters):
        size = (img_rows, img_cols)
        size = (size[0] - 8) // 2 + 1, (size[1] - 8) // 2 + 1
        size = (size[0] - 6) // 2 + 1, (size[1] - 6) // 2 + 1
        size = (size[0] - 5) // 1 + 1, (size[1] - 5) // 1 + 1
        return size[0] * size[1] * nb_filters * 2


class ODE(nn.Module):
    """
        odefunc can be any function, as long as its forward mapping takes t,x and
        outputs the same shape as x.
    """

    def __init__(self, odefunc, int_mtd='euler', device=None):
        super(ODE, self).__init__()
        self.odefunc = odefunc
        self.int_mtd = int_mtd
        self.device = _get_device(device)

    def forward(self, x, reverse=False):
        integration_times = torch.linspace(0, 1, 2, device=x.device, dtype=x.dtype)
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        predz = tdeq.odeint(self.odefunc, x, integration_times, method=self.int_mtd)
        return predz


class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        act = nn.ReLU()
        hid1, hid2, hid3 = 128, 256, 512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hid1, kernel_size=8, stride=2, padding=3),
            act,
            nn.Conv2d(hid1, hid2, kernel_size=6, stride=2, padding=0),
            act,
            nn.Conv2d(hid2, hid3, kernel_size=5, stride=1, padding=0),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hid3, hid2, kernel_size=5, stride=1, padding=0),
            act,
            nn.ConvTranspose2d(hid2, hid1, kernel_size=6, stride=2, padding=0),
            act,
            nn.ConvTranspose2d(hid1, 1, kernel_size=8, stride=2, padding=3)
        )

    def forward(self, t, x):
        return self.decoder(self.encoder(x))


class ImageAutoencoder(nn.Module):
    def __init__(self, channels=3):
        super(ImageAutoencoder, self).__init__()
        act = nn.ReLU()
        hid1, hid2, hid3 = 128, 256, 512
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hid1, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(hid1, hid2, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(hid2, hid3, kernel_size=3, stride=2, padding=1),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hid3, hid2, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            nn.ConvTranspose2d(hid2, hid1, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            nn.ConvTranspose2d(hid1, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, t, x):
        return self.decoder(self.encoder(x))


class FlexibleCNN(nn.Module):
    def __init__(self, img_rows=32, img_cols=32, channels=3, nb_filters=64, nb_classes=2):
        super(FlexibleCNN, self).__init__()
        self.activation = nn.ELU()
        self.features = nn.Sequential(
            nn.Conv2d(channels, nb_filters, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(nb_filters * 2 * 4 * 4, nb_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nb_classes=2):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, nb_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class TabularFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TabularFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, x):
        return self.net(x.view(x.size(0), -1))


def get_flowmodel(device=None, dtype=torch.float32, image_shape=(1, 28, 28),
                  input_dim=None, tabular_hidden_dim=128):
    device = _get_device(device)
    int_mtd = 'euler'
    if image_shape is None:
        if input_dim is None:
            raise ValueError("input_dim is required when image_shape is None.")
        odefunc = TabularFlow(input_dim, hidden_dim=tabular_hidden_dim)
    else:
        channels, img_rows, img_cols, _ = _normalize_image_shape(image_shape)
        if channels == 1 and img_rows == 28 and img_cols == 28:
            odefunc = MNISTAutoencoder()
        else:
            odefunc = ImageAutoencoder(channels=channels)
    model = ODE(odefunc, int_mtd, device=device)
    return model.to(device=device, dtype=dtype)


class FDROClassifier:
    def __init__(self, model, flow_model_ls, device, dtype=torch.float32,
                 num_classes=2, image_shape=(1, 28, 28)):
        self.model = model
        self.flow_model_ls = flow_model_ls
        self.device = device
        self.dtype = dtype
        self.num_classes = num_classes
        self.image_shape = image_shape

    def predict(self, X, batch_size=512):
        self.model.eval()
        X_tensor = _to_fdro_tensor(X, self.device, self.dtype, self.image_shape)
        preds = []
        with torch.no_grad():
            for start in range(0, X_tensor.size(0), batch_size):
                output = self.model(X_tensor[start:start + batch_size])
                _, predicted = torch.max(output.data, 1)
                preds.append(predicted.detach().cpu())
        return np.array(torch.cat(preds, dim=0).tolist())


def _frm(x, y, model, flow_model_ls, gamma, num_classes, loss_recorder=None):
    def get_loss1_and_2(predz, x_c, y_c):
        loss1 = criterion(model(predz[-1]), y_c)
        diff = (predz[-1] - x_c).view(x_c.size(0), -1)
        loss2 = 0.5 / gamma * torch.norm(diff, 2, 1).pow(2).mean()
        num_x = x_c.size(0)
        return num_x * loss1, num_x * loss2, loss1 - loss2

    criterion = nn.CrossEntropyLoss()
    loss1_1_ls, loss2_1_ls, loss_ls = [], [], []
    for c in range(num_classes):
        idx = y == c
        if idx.sum() == 0:
            loss1_1_ls.append(0)
            loss2_1_ls.append(0)
            loss_ls.append(None)
            continue
        predz = flow_model_ls[c](x[idx])
        loss1, loss2, loss = get_loss1_and_2(predz, x[idx], y[idx])
        loss1_1_ls.append(loss1)
        loss2_1_ls.append(loss2)
        loss_ls.append(loss)

    if loss_recorder is not None:
        num_x_tot = x.size(0)
        loss1, loss2 = sum(loss1_1_ls) / num_x_tot, sum(loss2_1_ls) / num_x_tot
        loss_recorder['loss_LFD_classifier'].append(float(loss1.item()))
        loss_recorder['loss_LFD_w2'].append(float(-loss2.item()))

    return loss_ls


def _frm_wrapper(full_X, full_y, model, flow_model_ls, optimizer_flow_ls, scheduler_ls,
                 batch_size, FRM_steps, gamma, num_classes, loss_recorder=None):
    data, target = full_X, full_y
    for _ in range(FRM_steps):
        rand_idx = torch.randperm(full_X.size(0), device=full_X.device)[:batch_size]
        data, target = full_X[rand_idx], full_y[rand_idx]
        loss_ls = _frm(data, target, model, flow_model_ls, gamma, num_classes, loss_recorder)
        for c in range(num_classes):
            if loss_ls[c] is None:
                continue
            optimizer_flow_ls[c].zero_grad()
            loss_ls[c].backward()
            optimizer_flow_ls[c].step()
            scheduler_ls[c].step()

    with torch.no_grad():
        target_ls = []
        perturbed_data_ls = []
        for c in range(num_classes):
            idx_c = target == c
            if idx_c.sum() == 0:
                continue
            target_ls.append(target[idx_c])
            perturbed_data_ls.append(flow_model_ls[c](data[idx_c])[-1])
        target = torch.cat(target_ls, dim=0)
        perturbed_data = torch.cat(perturbed_data_ls, dim=0)

    return perturbed_data, target


def train_FDRO(X1_train, X2_train, batch_tot=100, batch_size=512, gamma=5, FRM_steps=3,
               lr_flow=1e-4, lr_cnn=1e-4, device=None, dtype=torch.float32,
               seed=None, verbose=True, image_shape=(1, 28, 28),
               tabular_hidden_dim=128, classifier_pretrain_steps=0,
               early_stopping=False, tol=1e-3, min_batches=2):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = _get_device(device)
    X_train = np.concatenate([X1_train, X2_train], axis=0)
    y_train = np.concatenate([np.zeros(len(X1_train)), np.ones(len(X2_train))]).astype(np.int64)

    full_X = _to_fdro_tensor(X_train, device, dtype, image_shape)
    full_y = torch.tensor(y_train, dtype=torch.long, device=device)

    if image_shape is None:
        input_dim = full_X.size(1)
        model = TabularMLP(input_dim, hidden_dim=tabular_hidden_dim, nb_classes=2).to(device=device, dtype=dtype)
    else:
        input_dim = None
        channels, img_rows, img_cols, _ = _normalize_image_shape(image_shape)
        if channels == 1 and img_rows == 28 and img_cols == 28:
            model = CNN(nb_classes=2).to(device=device, dtype=dtype)
        else:
            model = FlexibleCNN(
                img_rows=img_rows,
                img_cols=img_cols,
                channels=channels,
                nb_classes=2,
            ).to(device=device, dtype=dtype)
    num_classes = 2
    flow_model_ls = [
        get_flowmodel(device, dtype, image_shape, input_dim=input_dim, tabular_hidden_dim=tabular_hidden_dim)
        for _ in range(num_classes)
    ]
    optimizer_flow_ls = [
        optim.Adam(flow_model.parameters(), lr=lr_flow, maximize=True)
        for flow_model in flow_model_ls
    ]
    scheduler_ls = [
        optim.lr_scheduler.StepLR(optimizer_flow, step_size=1, gamma=1)
        for optimizer_flow in optimizer_flow_ls
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_cnn)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    batch_size = min(batch_size, full_X.size(0))
    loss_recorder = {
        'loss_classifier': [],
        'loss_LFD_classifier': [],
        'loss_LFD_w2': [],
    }

    for _ in range(classifier_pretrain_steps):
        rand_idx = torch.randperm(full_X.size(0), device=full_X.device)[:batch_size]
        output = model(full_X[rand_idx])
        loss = criterion(output, full_y[rand_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_recorder['loss_classifier'].append(float(loss.item()))

    previous_loss = None
    converged = False
    batches_run = 0

    model.train()
    for batch in range(1, batch_tot + 1):
        perturbed_data, target = _frm_wrapper(
            full_X, full_y, model, flow_model_ls, optimizer_flow_ls, scheduler_ls,
            batch_size, FRM_steps, gamma, num_classes, loss_recorder
        )
        perturbed_output = model(perturbed_data)
        loss_adv = criterion(perturbed_output, target)
        optimizer.zero_grad()
        loss_adv.backward()
        loss_recorder['loss_classifier'].append(float(loss_adv.item()))
        optimizer.step()
        scheduler.step()

        if verbose and (batch % 100 == 0 or batch == batch_tot):
            print(f'After batch {batch}/{batch_tot}, loss = {loss_adv.item():.2e}')

        current_loss = float(loss_adv.item())
        batches_run = batch
        if early_stopping and previous_loss is not None and batch >= min_batches:
            relative_change = abs(current_loss - previous_loss) / max(abs(previous_loss), 1e-12)
            if relative_change <= tol:
                converged = True
                if verbose:
                    print(
                        f"Early stopping at batch {batch}/{batch_tot}, "
                        f"relative loss change = {relative_change:.2e}"
                    )
                break
        previous_loss = current_loss

    fdro_model = FDROClassifier(
        model,
        flow_model_ls,
        device,
        dtype=dtype,
        num_classes=num_classes,
        image_shape=image_shape,
    )
    fdro_model.loss_recorder = loss_recorder
    fdro_model.training_summary = {
        'batches_run': batches_run,
        'converged': converged,
    }
    return fdro_model

# Add testing functions for FDRO model. 
def test_FDRO(fdro_model, X_test_1, X_test_2, observation_size):
    X_test = np.concatenate([X_test_1, X_test_2], axis=0)
    preds = fdro_model.predict(X_test)

    preds_grouped = preds.reshape(-1, observation_size)
    group_votes = preds_grouped.mean(axis=1)
    group_preds = np.where(group_votes > 0.5, 1, 0)

    N_test_0 = len(X_test_1) // observation_size
    N_test_1 = len(X_test_2) // observation_size
    y_test_grouped = np.concatenate([np.zeros(N_test_0), np.ones(N_test_1)])

    risk = 1 - np.mean(y_test_grouped == group_preds)
    return risk

# Add testing functions for FDRO model (metrics: error rate, accuracy, f1-score)
def test_FDRO_with_metrics(fdro_model, X_test_1, X_test_2, observation_size):
    X_test = np.concatenate([X_test_1, X_test_2], axis=0)
    preds = fdro_model.predict(X_test)

    preds_grouped = preds.reshape(-1, observation_size)
    group_votes = preds_grouped.mean(axis=1)
    group_preds = np.where(group_votes > 0.5, 1, 0)

    N_test_0 = len(X_test_1) // observation_size
    N_test_1 = len(X_test_2) // observation_size
    y_test_grouped = np.concatenate([np.zeros(N_test_0), np.ones(N_test_1)])

    accuracy = np.mean(y_test_grouped == group_preds)
    return {
        'error_rate': 1 - accuracy,
        'accuracy': accuracy,
        'f1_score': float(f1_score(y_test_grouped, group_preds, zero_division=0)),
    }


def FDRO(X1_train, X2_train, **kwargs):
    return train_FDRO(X1_train, X2_train, **kwargs)
