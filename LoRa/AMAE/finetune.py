import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from model import Encoder, Decoder, Classifier
from get_dataset import FineTuneDataset_prepared
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(encoder,
          classifier,
          dataloader,
          optim_classifier,
          scheduler_classifier,
          epoch,
          device_num,
          writer
          ):
    encoder.eval()
    classifier.train()
    device = torch.device("cuda:" + str(device_num))
    loss_ce = 0
    correct = 0
    mse_loss = F.mse_loss
    for data_label in dataloader:
        data, target = data_label
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optim_classifier.zero_grad()

        z = encoder(data)
        logits = F.log_softmax(classifier(z))
        # target = np.squeeze(target, axis=1)
        # loss_ce_batch = F.nll_loss(logits, target)
        loss_ce_batch = F.nll_loss(logits, target)
        loss_ce_batch.backward()
        optim_classifier.step()
        scheduler_classifier.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss_ce += loss_ce_batch.item()


    loss_ce /= len(dataloader)

    fmt = 'Train Epoch: {} \tCE_Loss, {:.8f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            epoch,
            loss_ce,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )

    writer.add_scalar('CE_Loss/train', loss_ce, epoch)

def validation(encoder, classifier, test_dataloader, epoch, device_num, writer):
    encoder.eval()
    classifier.eval()
    loss_ce = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            z = encoder(data)
            logits = F.log_softmax(classifier(z))
            # target = np.squeeze(target, axis=1)
            loss_ce += F.nll_loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss_ce /= len(test_dataloader.dataset)
        fmt = '\nValidation set: CE_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
        print(
            fmt.format(
                loss_ce,
                correct,
                len(test_dataloader.dataset),
                100.0 * correct / len(test_dataloader.dataset),
            )
        )

        writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
        writer.add_scalar('Classifier_Loss/validation', loss_ce, epoch)

    return loss_ce

def Test(encoder, classifier, test_dataloader):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            target = target.squeeze()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)

            z = encoder(data)
            logits = F.log_softmax(classifier(z), dim=1)
            # target = np.squeeze(target, axis=1)
            test_loss += loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target)-1] = pred.tolist()
            target_real[len(target_real):len(target)-1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

        #target_pred = np.array(target_pred).reshape(1000)

        #target_real = np.array(target_real).reshape(1000)

    # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("TripleGAN_15label/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    # 将原始标签存下来

    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("TripleGAN_15label/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

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
    test_acc = 100.0 * correct / len(test_dataloader.dataset)
    return test_acc


def train_and_validation(encoder,
                         classifier,
                         dataloader,
                         optim_classifier,
                         scheduler_classifier,
                         epochs,
                         encoder_save_path,
                         classifier_save_path,
                         device_num,
                         writer):
    for epoch in range(1, epochs + 1):
        train(encoder,
              classifier,
              dataloader,
              optim_classifier,
              scheduler_classifier,
              epoch,
              device_num,
              writer)
    torch.save(encoder, encoder_save_path)
    torch.save(classifier, classifier_save_path)

class Config:
    def __init__(
            self,
            train_batch_size: int = 64,
            test_batch_size: int = 64,
            epochs: int = 100,
            lr_classifier: float = 0.001,
            n_classes: list = [20, 30],
            encoder_save_path: str = 'model_weight/MAE_encoder_IQ.pth',
            classifier_save_path: str = 'model_weight/MAE_classifier_IQ.pth',
            encoder_load_path: str = 'model_weight/pretrain_MAE_encoder_IQ.pth',
            device_num: int = 0,
            iteration: int = 100,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_classifier = lr_classifier
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.classifier_save_path = classifier_save_path
        self.encoder_load_path = encoder_load_path
        self.device_num = device_num
        self.iteration = iteration

def main(k, snr, seed):
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs_SimMIM_IQ")

    # RANDOM_SEED = 300  # any random number
    set_seed(seed)
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(range(conf.n_classes[0], conf.n_classes[1]), k, snr, seed)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    encoder = torch.load(conf.encoder_load_path)
    # encoder = torch.load("model_weight/pretrain_MAE_clustering_encoder_IQ.pth")
    classifier = Classifier()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = False

    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=conf.lr_classifier)

    scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)

    train_and_validation(encoder,
                         classifier,
                         train_dataloader,
                         optim_classifier,
                         scheduler_classifier,
                         conf.epochs,
                         conf.encoder_save_path,
                         conf.classifier_save_path,
                         conf.device_num,
                         writer)

    encoder = torch.load(conf.encoder_save_path)
    classifier = torch.load(conf.classifier_save_path)
    test_acc = Test(encoder, classifier, test_dataloader)
    return test_acc


if __name__ == '__main__':
    conf = Config()
    k = 20
    print(f'-------------------------k={k}------------------------------')
    for snr in [-10,-5,0,5,10]:
        print(f'-------------------------SNR={snr}------------------------------')
        test_acc_all = []
        for seed in range(30):
            seed += 2024
            print(f'-------------------------SEED={seed}------------------------------')
            test_acc = main(k, snr, seed)
            test_acc_all.append(test_acc)
            print(f"seed={seed},test_acc={test_acc}\n")
        df = pd.DataFrame(test_acc_all)
        df.to_excel(
        f"test_result/Frozen_PT_0-9_FT_{conf.n_classes[0]}-{conf.n_classes[1]-1}_{k}shot_SNR={snr}.xlsx")


