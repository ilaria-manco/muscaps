import os
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

from muscaps.utils.logger import Logger
from muscaps.trainers.muscaps_trainer import MusCapsTrainer
from muscaps.utils.utils import load_conf, merge_conf, get_root_dir
from muscaps.datasets.audiocaption import AudioCaptionDataset
from muscaps.models.cnn_lstm_caption import CNNLSTMCaption
from muscaps.models.cnn_attention_lstm import AttentionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a music captioning model")

    parser.add_argument("model", type=str, help="name of the model")

    parser.add_argument("--experiment_id", type=str,
                        help="experiment id under which checkpoint was saved", default=None)
    parser.add_argument("--config_path", type=str, help="path to base config file",
                        default=os.path.join(get_root_dir(), "configs", "default.yaml"))
    parser.add_argument("--dataset", type=str,
                        help="name of the dataset", default="audiocaption")
    parser.add_argument("--feature_extractor", type=str,
                        help="name of audio feature extraction mdoel", default=None)
    parser.add_argument("--pretrained_model", type=str,
                        help="version of pretrained model", default=None)
    parser.add_argument("--fusion", type=str,
                        help="fusion strategy", default=None)
    parser.add_argument("--finetune", type=str,
                        help="whether to finetune audio feature extractor", default=None)
    parser.add_argument("--device_num", type=str, default="0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    if params.experiment_id is None:
        # 1. Load config (base + dataset + model)
        base_conf = load_conf(params.config_path)

        if params.dataset == "audiocaption":
            dataset_conf_path = os.path.join(base_conf.env.base_dir,
                                             AudioCaptionDataset.config_path())
        else:
            print("{} dataset not supported".format(params.dataset))

        if params.model == "attention":
            model_conf_path = os.path.join(base_conf.env.base_dir,
                                           AttentionModel.config_path())
        elif params.model == "baseline":
            model_conf_path = os.path.join(base_conf.env.base_dir,
                                           CNNLSTMCaption.config_path())
        else:
            print("{} model not supported".format(params.model))

        config = merge_conf(params.config_path,
                            dataset_conf_path, model_conf_path)

        # Update config values with command line args if input
        if params.feature_extractor is not None:
            OmegaConf.update(config, "model_config.feature_extractor_type",
                             params.feature_extractor)
        if params.pretrained_model is not None:
            OmegaConf.update(config, "model_config.pretrained_version",
                             params.pretrained_model)
        if params.fusion is not None:
            OmegaConf.update(config, "model_config.fusion",
                             params.fusion)
        if params.finetune is not None:
            OmegaConf.update(config, "model_config.finetune",
                             params.finetune)
    else:
        config = OmegaConf.load(os.path.join(
            get_root_dir(), "save/experiments/{}/config.yaml".format(params.experiment_id)))

    logger = Logger(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    trainer = MusCapsTrainer(config, logger)
    trainer.train()
