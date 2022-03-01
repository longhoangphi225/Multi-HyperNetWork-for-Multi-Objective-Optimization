import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_trigonometric_dataset(n_samples=400, n_targets=2, cycles=1):
    X = np.random.uniform(0, cycles * 2 * np.pi, n_samples)
    Y = np.zeros((n_samples, n_targets))

    Y[:, 0] = np.cos(X)
    Y[:, 1] = np.sin(X)
    if n_targets==3:
        Y[:, 2] = np.sin(X + 1*np.pi)

    # needs extra dimension so that a NN recognizes these as a batch
    X = X[:, None]
    return X, Y


def train_and_val_split(data_x, data_y, train_ratio=0.5):
    nsamples = len(data_x)
    print("total data: ", nsamples)
    indices = np.arange(nsamples)
    np.random.shuffle(indices)

    train_indices = indices[:int(nsamples*train_ratio)]
    val_indices = indices[int(nsamples*train_ratio):]

    train_x, train_y = data_x[train_indices], data_y[train_indices]
    validation_x, validation_y = data_x[val_indices], data_y[val_indices]
    print("training data: {}, validation data: {}".format(
            len(train_x), len(validation_x)))
    return train_x, train_y, validation_x, validation_y


def load_datasets(target_device, cfg):
    n_samples = cfg["n_samples"]
    n_targets = cfg["n_mo_obj"]
    train_ratio = cfg["train_ratio"]
    data_x, data_y = generate_trigonometric_dataset(n_samples=n_samples, n_targets=n_targets)
    train_x, train_y, validation_x, validation_y = train_and_val_split(data_x, data_y, train_ratio=train_ratio)

    train_x = torch.from_numpy(train_x).float().to(target_device)
    train_y = torch.from_numpy(train_y).float().to(target_device)
    validation_x = torch.from_numpy(validation_x).float().to(target_device)
    validation_y = torch.from_numpy(validation_y).float().to(target_device)
    return train_x, train_y, validation_x, validation_y


class Net(nn.Module):
    def __init__(self, target_device):
        super().__init__()
        target_hidden_dim = 50
        outdim = 100
        self.mlp = nn.Sequential(
          nn.Linear(1, target_hidden_dim),
          nn.Linear(target_hidden_dim, outdim)
        )
        self.target_device = target_device
    def forward(self):
        x = torch.ones([1, 1], dtype=torch.float32).to(self.target_device)
        x = self.mlp(x)
        return x


class ScaledMSELoss(nn.Module):
    """mse loss scaled by 0.01"""
    def __init__(self, reduction='none'):
        super(ScaledMSELoss, self).__init__()
        self.reduction = reduction


    def forward(self, inputs, target):
        """
        out = 0.01 * mse_loss(inputs, target)
        """
        out = 0.01 * torch.nn.functional.mse_loss(inputs, target, reduction=self.reduction) 
        return out

def toy_loss_1(output):
  d = output.shape[1]
  d = torch.from_numpy(np.array(d, dtype='float32'))
  return torch.mean(1-torch.exp(-torch.norm(output-1/torch.sqrt(d), p=2)))

def toy_loss_2(output):
  d = output.shape[1]
  d = torch.from_numpy(np.array(d, dtype='float32'))
  return torch.mean(1-torch.exp(-torch.norm(output+1/torch.sqrt(d), p=2)))
      
class Loss(nn.Module):
    """Evaluation of two losses"""
    def __init__(self):
        super(Loss, self).__init__()
        #self.implemented_loss = ["MSELoss", "L1Loss", "ScaledMSELoss"]

        self.loss_list = [toy_loss_1, toy_loss_2]

        """for loss_name in loss_name_list:
            if loss_name not in self.implemented_loss:
                raise NotImplementedError("{} not implemented. Implemented losses are: {}".format(loss_name, self.implemented_loss))
            elif loss_name == "MSELoss":
                self.loss_list.append( torch.nn.MSELoss(reduction='none') )
            elif loss_name == "L1Loss":
                self.loss_list.append( torch.nn.L1Loss(reduction='none') )
            elif loss_name == "ScaledMSELoss":
                self.loss_list.append( ScaledMSELoss(reduction='none') )"""



    def forward(self, inputs):
        """
        out_list = list of losses, where each loss is a tensor of losses for each sample
        """
        #assert(target.shape[1] == len(self.loss_list))
        #target = target.to(inputs.device)
        out_list = []
        for i, loss_fn in enumerate(self.loss_list):
            out = loss_fn(inputs)
            out_list.append(out.view(-1))

        out = torch.stack(out_list, dim=0)
        return out


def initialize_losses(cfg):
    #loss_names = cfg["loss_names"]
    loss_fn = Loss()
    return loss_fn


if __name__ == '__main__':
    pass

