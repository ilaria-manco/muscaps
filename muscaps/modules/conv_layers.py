""" Code from https://github.com/minzwon/sota-music-tagging-models/ 

MIT License

Copyright (c) 2020 Music Technology Group, Universitat Pompeu Fabra. Code developed by Minz Won.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import librosa
import torch
import torch.nn as nn


class Conv_1d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 shape=3,
                 stride=1,
                 pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels,
                              output_channels,
                              shape,
                              stride=stride,
                              padding=shape // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 shape=3,
                 stride=1,
                 pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              shape,
                              stride=stride,
                              padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_V(nn.Module):
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              filter_shape,
                              padding=(0, filter_shape[1] // 2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out


class Conv_H(nn.Module):
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = nn.Conv1d(input_channels,
                              output_channels,
                              filter_length,
                              padding=filter_length // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out


class Res_2d_mp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res_2d_mp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out


# Modules for harmonic filters
def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0**((midi - 69.0) / 12.0))


def note_to_midi(note):
    return librosa.core.note_to_midi(note)


def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi('C1')

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level
