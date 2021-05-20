import os
import torch
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import argparse

from muscaps.scripts.eval import Evaluation
from muscaps.utils.utils import get_root_dir
from muscaps.utils.text_decoder import GreedyDecoder, BeamSearchDecoder
from muscaps.utils.logger import Logger
from muscaps.caption_evaluation_tools import eval_metrics


class Captioning(Evaluation):
    def __init__(self, config, logger, experiment_id):
        super().__init__(config, logger, experiment_id)

        if self.config.model_config.inference_type == "greedy":
            self.text_decoder = GreedyDecoder(
                self.vocab, self.config)
            self.predictions_path = os.path.join(
                self.logger.experiment_dir, "predictions.json")
        elif self.config.model_config.inference_type == "beam_search":
            self.text_decoder = BeamSearchDecoder(
                self.vocab, self.config)
            self.predictions_path = os.path.join(
                self.logger.experiment_dir, "predictions_beam_{}.json".format(self.config.model_config.beam_size))

    def predict_caption(self, audio, audio_len):
        audio = audio.float().to(device=self.device)
        predicted_caption = self.text_decoder.decode(
            self.model, audio, audio_len)

        return predicted_caption

    def obtain_predictions(self, save_predictions=True):
        if os.path.exists(self.predictions_path):
            predictions, true_captions, audio_paths = json.load(
                open(self.predictions_path)).values()
        else:
            predictions = []
            true_captions = []
            audio_paths = []

            self.logger.write("Predicting captions")

            with torch.no_grad():
                for i, batch in enumerate(tqdm(self.test_loader)):
                    audio, audio_len, true_caption, x_len = batch

                    # predict and decode caption
                    pred_caption = self.predict_caption(audio, audio_len)

                    pred_caption_decoded = self.text_decoder.decode_caption(
                        pred_caption)
                    predictions.append(pred_caption_decoded)

                    # decode target caption
                    true_caption_decoded = self.text_decoder.decode_caption(
                        true_caption.cpu().tolist()[0])
                    true_captions.append([true_caption_decoded])

                    audio_path = self.test_loader.dataset.audio_paths[i]
                    audio_paths.append(audio_path)

            if save_predictions:
                with open(self.predictions_path, 'w') as outfile:
                    output = {
                        "predictions": predictions, "true_captions": true_captions, "audio_paths": audio_paths}
                    json.dump(output, outfile)

        return predictions, true_captions, audio_paths

    def compute_metrics(self):
        predictions, true_captions, _ = self.obtain_predictions()
        metrics, _ = eval_metrics.evaluate_metrics_from_lists(
            predictions, true_captions)
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a music captioning model")

    parser.add_argument("experiment_id", type=str)
    parser.add_argument("--metrics", type=bool, default=False)
    parser.add_argument("--device_num", type=str, default="0")
    parser.add_argument("--decoding", type=str,
                        help="type of decoding to use in inference", default=None)
    parser.add_argument("--beam_size", type=int,
                        help="beam size to use in beam search decoding", default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    experiment_id = params.experiment_id
    config = OmegaConf.load(os.path.join(
        get_root_dir(), "save/experiments/{}/config.yaml".format(experiment_id)))

    if params.decoding is not None:
        OmegaConf.update(
            config, "model_config.inference_type", params.decoding)
    if params.beam_size is not None:
        OmegaConf.update(
            config, "model_config.beam_size", params.beam_size)

    logger = Logger(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num
    evaluation = Captioning(config, logger, experiment_id)
    if params.metrics:
        evaluation.compute_metrics()
