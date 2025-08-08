import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import _create_model_training_folder, Data_augment
from torch.utils.data.dataloader import DataLoader
import copy

class SimCLR:
    def __init__(self, model, optimizer,  **params):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = params["batch_size"]
        self.temperature = params["temperature"]
        self.epochs = params["max_epochs"]
        self.writer = SummaryWriter(f"runs/seed300_pretrain_{params['ft']}ft_class_in_{params['class_start']}-{params['class_end']}")
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "run.py", 'simclr.py'])

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, epoch_counter):
        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False,
                               slice=False, isAdvAug=True)
        self.model.train()
        train_loss_epoch = 0
        for batch_view, _ in train_loader:
            batch_view = batch_view.cuda()
            batch_view_1 = Augment(batch_view, online_network=self.model)
            batch_view_2 = Augment(batch_view, online_network=self.model)
            batch_data = torch.cat([batch_view_1, batch_view_2], dim=0)
            batch_data = batch_data.cuda()

            features = self.model(batch_data)[1]
            self.optimizer.zero_grad()
            logits, labels = self.info_nce_loss(features)
            train_loss_batch = self.criterion(logits, labels)
            train_loss_batch.backward()
            self.optimizer.step()
            train_loss_epoch += train_loss_batch.item()

        train_loss_epoch /= len(train_loader)
        self.writer.add_scalar('train_loss_epoch', train_loss_epoch, global_step=epoch_counter)

    def eval(self, val_loader, epoch_counter):
        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False,
                               slice=False, isAdvAug=True)
        self.model.eval()
        eval_loss_epoch = 0
        for batch_view, _ in val_loader:
            batch_view = batch_view.cuda()
            batch_view_1 = Augment(batch_view, online_network=self.model)
            batch_view_2 = Augment(batch_view, online_network=self.model)
            batch_data = torch.cat([batch_view_1, batch_view_2], dim=0)
            batch_data = batch_data.cuda()
            with torch.no_grad():
                features = self.model(batch_data)[1]
                logits, labels = self.info_nce_loss(features)
                eval_loss_batch = self.criterion(logits, labels)
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
        for epoch_counter in range(self.epochs):
            print(f'Epoch={epoch_counter}')
            self.train(train_loader, epoch_counter)
            eval_loss_epoch = self.eval(val_loader, epoch_counter)
            if eval_loss_epoch <= loss_min:
                torch.save(self.model, os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = eval_loss_epoch
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        torch.save(self.model, os.path.join(model_checkpoints_folder, 'model.pth'))
