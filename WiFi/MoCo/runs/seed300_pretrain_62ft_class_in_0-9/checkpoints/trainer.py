import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from utils import _create_model_training_folder, Data_augment
from MoCo_builder import MoCo

class SimSiamTrainer:
    def __init__(self, device, **params):
        self.online_network = MoCo(dim=128, K=1280, m=params['m'], T=0.07, mlp=False)
        self.device = device
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter(f"runs/seed300_pretrain_{params['ft']}ft_class_in_{params['class_start']}-{params['class_end']}")
        self.batch_size = params['batch_size']
        self.lr = params['lr']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    def train(self, train_loader, epoch_counter):
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.online_network.to(self.device)
        self.online_network.train()

        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False,
                               slice=False, isAdvAug=True)
        train_loss_epoch = 0
        for batch_view, _ in train_loader:
            batch_view = batch_view.to(self.device)
            batch_view_1 = Augment(batch_view, online_network=self.online_network.encoder_q)
            batch_view_2 = Augment(batch_view, online_network=self.online_network.encoder_q)
            self.optimizer.zero_grad()
            train_loss_batch = self.update(batch_view_1, batch_view_2)
            train_loss_batch.backward()
            self.optimizer.step()
            train_loss_epoch += train_loss_batch.item()
        train_loss_epoch /= len(train_loader)
        self.writer.add_scalar('train_loss_epoch', train_loss_epoch, global_step=epoch_counter)

    def eval(self, val_loader, epoch_counter):
        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False,
                               slice=False, isAdvAug=True)
        self.online_network.eval()
        eval_loss_epoch = 0
        for batch_view, _ in val_loader:
            batch_view = batch_view.to(self.device)
            batch_view_1 = Augment(batch_view, online_network=self.online_network.encoder_q)
            batch_view_2 = Augment(batch_view, online_network=self.online_network.encoder_q)
            with torch.no_grad():
                eval_loss_batch = self.update(batch_view_1, batch_view_2)
                eval_loss_epoch += eval_loss_batch.item()
        eval_loss_epoch /= len(val_loader)
        self.writer.add_scalar('eval_loss_epoch', eval_loss_epoch, global_step=epoch_counter)
        print(f"The loss on eval dataset: {eval_loss_epoch}")
        return eval_loss_epoch

    def train_and_val(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  drop_last=False, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  drop_last=False, shuffle=True)
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        loss_min = 10000000
        for epoch_counter in range(self.max_epochs):
            print(f'Epoch={epoch_counter}')
            self.train(train_loader, epoch_counter)
            eval_loss_epoch = self.eval(val_loader, epoch_counter)
            if eval_loss_epoch <= loss_min:
                torch.save(self.online_network.encoder_q, os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = eval_loss_epoch
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        torch.save(self.online_network.encoder_q, os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        output, target = self.online_network(im_q=batch_view_1, im_k=batch_view_2)
        loss = self.criterion(output, target)
        return loss
