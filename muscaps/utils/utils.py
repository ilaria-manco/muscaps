import os
import json
from omegaconf import OmegaConf
import torch


def load_conf(path_to_yaml):
    """ Wrapper for configuration file loading through OmegaConf. """
    conf = OmegaConf.load(path_to_yaml)
    if "env" in conf.keys():
        OmegaConf.update(conf, "env.base_dir", get_root_dir())
    return conf


def merge_conf(base_conf_path, dataset_conf_path, model_conf_path):
    """ Wrapper to merge multiple config files through OmegaConf. """
    base_conf = load_conf(base_conf_path)
    dataset_conf = load_conf(dataset_conf_path)
    model_conf = load_conf(model_conf_path)

    conf = OmegaConf.merge(
        base_conf, dataset_conf, model_conf)
    return conf


def save_json(output_path, content):
    with open(output_path, 'w') as outfile:
        json.dump(content, outfile)


def get_root_dir():
    root = os.path.dirname(
        os.path.abspath(__file__))
    root = os.path.abspath(
        os.path.join(root, "../.."))
    return root


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    Code from https://github.com/allenai/allennlp.

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(
            vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(
                vector * mask, dim=dim)
            result = result * mask
            result = result / \
                (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(
                masked_vector, dim=dim)
    return result
