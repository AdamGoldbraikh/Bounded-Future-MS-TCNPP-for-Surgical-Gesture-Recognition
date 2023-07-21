from DataStructures import ModelRecreate  # imports from our code
from FrameGenrator import FrameGenerator

import time  # for prints
from termcolor import colored
import tqdm

import math  # for functions
from torchvision import transforms
import torch

from typing import Callable  # make code look better

import warnings  # silence some warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extraction_examples(extractor, shape: tuple = None, num_examples=30):
    """
    builds a torch.jit._script.RecursiveScriptModule model that replicates the extractor and
    preforms 'num_examples' iterations of passing a random input of shape 'shape' to it to make it faster. 
    Can only do it with a pre-defined shape (the shape of each frame) 
    Args:
        extractor (EfficientNetV2): The extractor
        shape (tuple, optional): Defaults to None.
        num_examples (int, optional): number of iterations as described . Defaults to 30.
    Returns:
        torch.jit._script.RecursiveScriptModule model: as described
    """
    if shape is not None:
        print(colored("start examples of feature extraction", "blue"))
        f = torch.randn(1, 3, *shape, device=device)
        with torch.no_grad():
            extractor = extractor.to(device)
            extractor = torch.jit.script(
                extractor).to(device)
            extractor = torch.jit.freeze(extractor)
        for i in range(num_examples):
            extractor(f)
        print(colored("end examples of feature extraction", "blue"))
    return extractor.to(device)


def run(frame_gen, model, extractor, normalize: Callable, val_augmentation: Callable, use_extractions: bool = False, shape: tuple = None) -> list:
    """
    Runs the real online inference.

    Args:
        frame_gen : An object that yields frames 
        model : An MSTCN++ model to recreate.
        extractor : A model that takes a frame (as a tensor) and converts it to an embedding. 
        normalize (Callable, optional): does the normaliztion of the tensor frame after converting to tensor. If None, use Identity.
        val_augmentation (Callable, optional): the augmentations to the frame before converting to tensor. If None, use Identity. 
        use_extractions (bool, optional): If true, converts the extractor to a onednn model using examples. Makes the code faster but requires the shape of the frames. Defaults to True.
        shape (tuple, optional): shape of the frames after the augmentations. Defaults to None.

    Returns:
        list: list of torch tensors. list[i] is the output of the model for frame i.
    """
    if use_extractions:
        if shape is None:
            raise ValueError(
                "To use extractions, the shape needs to be not None")
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("The shapes needs to be a tuple of size 2")
        extractor = extraction_examples(extractor, shape=shape)

    val_augmentation = val_augmentation if val_augmentation is not None else (
        lambda x: x)
    normalize = normalize if normalize is not None else (lambda x: x)

    print(colored("initialize Model recreate - ready to start streaming", "yellow"))
    t0 = time.time()
    # with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    mr = ModelRecreate(model, model.w_max,
                       frame_gen, extractor, normalize, val_augmentation)

    num_layers = model.PG.num_layers
    frames = model.num_R * sum(min(model.w_max, 2**i) for i in range(num_layers))\
        + sum(min(model.w_max, max(2**i, 2**(num_layers-i-1)))
              for i in range(num_layers))
    took = time.time()-t0
    print(colored(
        f"finished initializing Model recreate, {frames} frames took {took} seconds ({frames/took} fps)", "yellow"))
    pbar = tqdm.tqdm(total=math.inf, unit=' frames')
    while True:
        try:
            next = mr.next()
            pbar.update(1)
            yield next

        except ValueError:
            pbar.bar_format = 'Total number of {n} frames. Calculated at avg of {rate_fmt} '
            pbar.close()
            return
