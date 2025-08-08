import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
import random
import numpy as np
from models.encoder_and_projection import Encoder_and_projection
from simclr import SimCLR
from get_dataset import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

RANDOM_SEED = 300 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_tr = config["trainer"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    X_train, Y_train = PreTrainDataset_prepared()
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=30)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    model = Encoder_and_projection().to(device)

    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["params"]["lr"])

    simclr = SimCLR(model=model, optimizer=optimizer, **config['trainer'])
    simclr.train_and_val(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
