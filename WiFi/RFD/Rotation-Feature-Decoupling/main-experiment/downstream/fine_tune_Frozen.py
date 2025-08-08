import os
import sys
print(sys.path)
sys.path.insert(0, 'models')
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from get_dataset import FineTuneDataset_prepared
from model_complexcnn import CVCNN as Encoder
from model_complexcnn import Classifier
from utils import set_log_file_handler, get_pretrain_encoder
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random
from config.config import get_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

os.environ['CUDA_VISIBLE_DEVICES']='3'

def train(online_network, classifier, loss_nll, train_dataloader, optimizer_classifier, scheduler_classifier, epoch, device, writer):
    online_network.eval()  # 启动训练, 允许更新模型参数
    classifier.train()
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optimizer_classifier.zero_grad()

        features = online_network(data)
        output = F.log_softmax(classifier(features), dim=1)
        nll_loss_batch = loss_nll(output, target)
        nll_loss_batch.backward()

        optimizer_classifier.step()
        scheduler_classifier.step()

        nll_loss += nll_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader)

    print('Train Epoch: {} \tClass_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', nll_loss, epoch)

def evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer):
    online_network.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = classifier(online_network(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/val', 100.0 * correct / len(val_dataloader.dataset), epoch)
    writer.add_scalar('Loss/val', test_loss, epoch)
    return test_loss

def test(online_network, classifier, test_dataloader, device):
    online_network.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            output = classifier(online_network(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

def train_and_test(online_network, classifier, loss_nll, train_dataloader, optim_classifier, scheduler_classifier, epochs, save_path_classifier, device, writer):
    for epoch in range(1, epochs + 1):
        train(online_network, classifier, loss_nll, train_dataloader, optim_classifier, scheduler_classifier, epoch, device, writer)
    torch.save(classifier, save_path_classifier)

def run(train_dataloader, test_dataloader, epochs, save_path_classifier, device, writer):
    config = get_config("./config/config.yaml")
    logger = set_log_file_handler(f"logs/{config['full_name']}")
    print(f"Training with: {device}")

    params = config['trainer']
    pretrained_path = params['pretrained_path']

    encoder = Encoder()
    encoder = get_pretrain_encoder(encoder, logger, pretrained_path)
    classifier = Classifier(6)

    if torch.cuda.is_available():
        online_network = encoder.to(device)
        classifier = classifier.to(device)

    for name, parameter in online_network.named_parameters():
        parameter.requires_grad = False

    loss_nll = nn.NLLLoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)

    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=0.001)

    scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)

    train_and_test(online_network, classifier, loss_nll, train_dataloader=train_dataloader, optim_classifier=optim_classifier, scheduler_classifier=scheduler_classifier, epochs=epochs, save_path_classifier=save_path_classifier, device=device, writer=writer)
    print("Test_result:")
    classifier = torch.load(save_path_classifier)
    test_acc = test(online_network, classifier, test_dataloader, device)
    return test_acc

def main(k, SEED, test_acc_all, snr):
    device = torch.device("cuda:0")

    print(f"SEED: {SEED}--------------------------------------------------------")
    set_seed(SEED)
    writer = SummaryWriter(f"./log_finetune/Frozen_PT_62ft_0-9_FT_62ft_10-15_{k}shot_{SEED}")

    save_path_classifier = f"./model_weight/Frozen_classifier_PT_62ft_0-9_FT_62ft_10-15_{k}shot_{SEED}.pth"

    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(62, range(10,16), k, snr, SEED)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # train
    test_acc = run(train_dataloader, test_dataloader, epochs=100, save_path_classifier=save_path_classifier, device=device, writer=writer)
    test_acc_all.append(test_acc)
    writer.close()


if __name__ == '__main__':
    k = 20
    for snr in [-10,-5,0,5,10]:
        test_acc_all = []
        for seed in range(30):
            print(f'-------------------------k={20}------------------------------')
            main(k, 2024 + seed, test_acc_all, snr)
        df = pd.DataFrame(test_acc_all)
        df.to_excel(
        f"test_result/Frozen_PT_62ft_0-9_FT_62ft_10-15_{k}shot_SNR={snr}.xlsx")

