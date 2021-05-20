import torch


class TextDecoder():
    def __init__(self, vocab, config):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()
        self._config = config.model_config
        self._device = torch.device(config.training.device)
        self.max_steps = self._config.max_caption_len

    def decode_caption(self, caption):
        decoded = [self._vocab.idx2word[i] for i in caption]
        decoded = ' '.join(decoded)
        return decoded


class GreedyDecoder(TextDecoder):
    def __init__(self, vocab, config):
        super().__init__(vocab, config)

    def reset(self):
        pass

    def decode(self, model, audio, audio_len):
        # Tensor to store all with previous words at each step; <sos> at t=0
        predicted_caption = torch.LongTensor(
            [self._vocab.get_id('<sos>')]).to(self._device)

        for t in range(self.max_steps):
            cap_lens = torch.Tensor(
                [len(predicted_caption)]).int().to(self._device)
            out = model(audio, audio_len, predicted_caption.unsqueeze(0),
                        cap_lens)
            prediction = torch.log_softmax(out, dim=2)
            prediction = torch.argmax(prediction, dim=2)
            prediction = prediction[:, -1]
            predicted_caption = torch.cat([predicted_caption, prediction])
            if prediction == self._vocab.word2idx['<eos>']:
                break
        return predicted_caption


class BeamSearchDecoder(TextDecoder):
    """ Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py"""

    def __init__(self, vocab, config):
        super().__init__(vocab, config)
        self.reset()

    def reset(self):
        self._beam_size = self._config.beam_size

        self._complete_seqs = []
        self._complete_seqs_scores = []
        self.k_prev_words = torch.LongTensor(
            [[self._vocab.get_id('<sos>')]] * self._beam_size).to(self._device)
        self.seqs = self.k_prev_words.to(self._device)
        self.top_k_scores = torch.zeros(self._beam_size, 1).to(self._device)

    def init_batch(self, audio, audio_len):
        cap_lens = torch.Tensor(
            [self.seqs.size(1)] * self._beam_size).int().to(self._device)
        audio = torch.cat(self._beam_size * [audio], dim=0)
        audio_len = torch.cat(self._beam_size * [audio_len], dim=0)

        return cap_lens, audio, audio_len

    def add_next_word(self, prev_word_inds, next_word_inds):
        seqs = torch.cat([self.seqs[prev_word_inds],
                          next_word_inds.unsqueeze(1)], dim=1)
        return seqs

    def decode(self, model, audio_in, audio_len_in):
        self.reset()
        cap_lens, audio, audio_len = self.init_batch(audio_in, audio_len_in)

        for t in range(self.max_steps):
            scores = model(audio, audio_len, self.seqs, cap_lens)
            scores = torch.log_softmax(scores, dim=2)[:, -1, :]

            scores = self.top_k_scores.expand_as(
                scores) + scores

            if t == 0:
                self.top_k_scores, self.top_k_words = scores[0].topk(
                    self._beam_size, 0, True, True)
            else:
                self.top_k_scores, self.top_k_words = scores.view(
                    -1).topk(self._beam_size, 0, True, True)

            prev_word_inds = self.top_k_words // self._vocab_size
            next_word_inds = self.top_k_words % self._vocab_size

            self.seqs = self.add_next_word(prev_word_inds, next_word_inds)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self._vocab.word2idx['<eos>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                self._complete_seqs.extend(
                    self.seqs[complete_inds].tolist())
                self._complete_seqs_scores.extend(
                    self.top_k_scores[complete_inds])
            self._beam_size -= len(complete_inds)

            if self._beam_size == 0:
                break
            self.seqs = self.seqs[incomplete_inds]
            cap_lens, audio, audio_len = self.init_batch(
                audio_in, audio_len_in)
            self.top_k_scores = self.top_k_scores[incomplete_inds].unsqueeze(1)
            self.top_k_prev_words = next_word_inds[incomplete_inds].unsqueeze(
                1)

            t += 1

        if len(self._complete_seqs_scores) > 0:
            i = self._complete_seqs_scores.index(
                max(self._complete_seqs_scores))
            seq = self._complete_seqs[i]
        else:
            seq = []

        return seq
