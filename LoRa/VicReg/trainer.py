import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from losses import vicreg_loss_func
from utils import _create_model_training_folder, Data_augment

class VicRegTrainer:
    def __init__(self, online_network, optimizer, device, **params):
        self.online_network = online_network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter(f"runs/seed300_pretrain_class_in_{params['class_start']}-{params['class_end']}")
        self.m = params['m']
        self.batch_size = params['batch_size']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    def train(self, train_loader, epoch_counter):
        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False,
                               slice=False, isAdvAug=True)
        self.online_network.train()
        train_loss_epoch = 0
        for batch_view, _ in train_loader:
            batch_view = batch_view.to(self.device)
            batch_view_1 = Augment(batch_view, online_network=self.online_network)
            batch_view_2 = Augment(batch_view, online_network=self.online_network)
            self.optimizer.zero_grad()
            z1 = self.online_network(batch_view_1)[1]
            z2 = self.online_network(batch_view_2)[1]
            train_loss_batch = vicreg_loss_func(z1, z2)
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
            batch_view_1 = Augment(batch_view, online_network=self.online_network)
            batch_view_2 = Augment(batch_view, online_network=self.online_network)
            with torch.no_grad():
                z1 = self.online_network(batch_view_1)[1]
                z2 = self.online_network(batch_view_2)[1]
                eval_loss_batch = vicreg_loss_func(z1, z2)
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
                torch.save(self.online_network, os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = eval_loss_epoch
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        torch.save(self.online_network, os.path.join(model_checkpoints_folder, 'model.pth'))
