import os
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from get_dataset import FineTuneDataset_prepared
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random
from math import pi
from math import cos
from math import floor

# RANDOM_SEED = 300 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser(description='PyTorch Complex_test Training')
parser.add_argument('--lr_encoder', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--lr_classifier', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
args = parser.parse_args(args=[])

def test(online_network, classifier, test_dataloader, device):
    online_network.eval()  # 启动验证，不允许更新模型参数
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
            output = classifier(online_network(data)[0])
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

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']

    device = torch.device("cuda:0")

    test_acc_all = []

    for i in range(10):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(f"./log_finetune/nofrozen_and_onelinear/pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot")

        save_path_classifier = f"./model_weight/nofrozen_and_onelinear/classifier_pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_online_network = f"./model_weight/nofrozen_and_onelinear/online_network_pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"

        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(config_ft['k_shot'])

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

        print("Test_result:")
        online_network = torch.load(save_path_online_network).cuda()
        classifier = torch.load(save_path_classifier).cuda()
        test_acc = test(online_network, classifier, test_dataloader, device)
        print(test_acc)

        test_acc_all.append(test_acc)
        writer.close()

    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"test_result/nofrozen_and_onelinear/pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx")

if __name__ == '__main__':
   main()