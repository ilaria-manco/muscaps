import os
import time
import numpy as np
from omegaconf import OmegaConf

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, Adadelta

from muscaps.datasets.base_dataset import custom_collate_fn
from muscaps.datasets.audiocaption import AudioCaptionDataset
from muscaps.models.cnn_lstm_caption import CNNLSTMCaption
from muscaps.models.cnn_attention_lstm import AttentionModel
from muscaps.trainers.base_trainer import BaseTrainer


class MusCapsTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.logger = logger
        self.device = torch.device(self.config.training.device)
        self.patience = self.config.training.patience
        self.lr = self.config.training.lr

        self.load_dataset()
        self.build_model()
        self.build_loss()
        self.build_optimizer()

    def load_dataset(self):
        self.logger.write("Loading dataset")
        dataset_name = self.config.dataset_config.dataset_name
        if dataset_name == "audiocaption":
            train_dataset = AudioCaptionDataset(self.config.dataset_config)
            val_dataset = AudioCaptionDataset(
                self.config.dataset_config, "val")
        else:
            raise ValueError(
                "{} dataset is not supported.".format(dataset_name))
        self.vocab = train_dataset.vocab
        self.logger.save_vocab(self.vocab.token_freq)
        OmegaConf.update(self.config, "model_config.vocab_size",
                         self.vocab.size)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=True)
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            collate_fn=custom_collate_fn)
        self.logger.write("Number of training samples: {}".format(
            train_dataset.__len__()))

    def build_model(self):
        self.logger.write("Building model")
        model_name = self.config.model_config.model_name
        if model_name == "cnn_lstm_caption":
            self.model = CNNLSTMCaption(
                self.config.model_config, self.vocab, self.device, teacher_forcing=True)
        elif model_name == "cnn_attention_lstm":
            self.model = AttentionModel(
                self.config.model_config, self.vocab, self.device, teacher_forcing=True)
        else:
            raise ValueError("{} model is not supported.".format(model_name))
        if self.model.audio_encoder.pretrained_version is not None and not self.model.finetune:
            for param in self.model.audio_encoder.feature_extractor.parameters():
                param.requires_grad = False
        self.model.to(self.device)

    def count_parameters(self):
        """ Count trainable parameters in model. """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def build_loss(self):
        self.logger.write("Building loss")
        loss_name = self.config.model_config.loss
        if loss_name == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(ignore_index=self.vocab.PAD_INDEX)
        else:
            raise ValueError("{} loss is not supported.".format(loss_name))
        self.loss = self.loss.to(self.device)

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_name = self.config.training.optimizer
        if optimizer_name == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == "adadelta":
            self.optimizer = Adadelta(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(
                "{} optimizer is not supported.".format(optimizer_name))

    def train(self):
        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write("Resumed training experiment with id {}".format(
                self.logger.experiment_id))
            self.load_ckp(self.logger.checkpoint_path)
        else:
            self.logger.write("Started training experiment with id {}".format(
                self.logger.experiment_id))
            self.start_epoch = 0

        # Adaptive learning rate
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=self.patience,
                                                   verbose=True)

        k_patience = 0
        best_val = np.Inf

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(
                self.train_loader, self.device, is_training=True)
            val_loss = self.train_epoch_val(
                self.val_loader, self.device, is_training=False)

            # Decrease the learning rate after not improving in the validation set
            scheduler.step(val_loss)

            # check if val loss has been improving during patience period. If not, stop
            is_val_improving = scheduler.is_better(val_loss, best_val)
            if not is_val_improving:
                k_patience += 1
            else:
                k_patience = 0
            if k_patience > self.patience * 2:
                print("Early Stopping")
                break

            best_val = scheduler.best

            epoch_time = time.time() - epoch_start_time
            lr = self.optimizer.param_groups[0]['lr']

            self.logger.update_training_log(epoch + 1, train_loss, val_loss,
                                            epoch_time, lr)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            # save checkpoint in appropriate path (new or best)
            self.logger.save_checkpoint(state=checkpoint,
                                        is_best=is_val_improving)

    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, data_loader, device, is_training):
        out_list = []
        target_list = []
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
            if self.model.audio_encoder.pretrained_version is not None:
                for module in self.model.audio_encoder.feature_extractor.modules():
                    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                        module.eval()
        else:
            self.model.eval()

        for i, batch in enumerate(data_loader):
            audio, audio_len, x, x_len = batch
            target_list.append(x)
            audio = audio.float().to(device=device)
            x = x.long().to(device=device)
            audio_len.to(device=device)
            out = self.model(audio, audio_len, x, x_len)

            out_list.append(out)

            target = x[:, 1:]  # target excluding sos token
            out = out.transpose(1, 2)
            loss = self.loss(out, target)

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.training.clip_gradients:
                    clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

            running_loss += loss.item()

            n_batches += 1

        return running_loss / n_batches

    def train_epoch_val(self, data_loader, device, is_training=False):
        with torch.no_grad():
            loss = self.train_epoch(data_loader, device, is_training=False)
        return loss
