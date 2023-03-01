import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torchvision.transforms.functional import pad, rotate, affine, InterpolationMode
import torch.nn.functional as F
from PIL import Image
import random
from typing import List

from model import ModelArgs

class MNISTData(MNIST):
    def __init__(self, args: ModelArgs, root: str, train, transform: List = [ToTensor()]):
        super(MNISTData, self).__init__(root, train, download=True)

        self.nearest = InterpolationMode.NEAREST
        self.operators = (
            lambda _: rotate(_, 60, self.nearest),  # rotate_left
            lambda _: rotate(_, -60, self.nearest),  # rotate_right
            lambda _: affine(_, 0, [0, -.15 * _.size(-2)], 1, [0, 0], self.nearest),  # translate_up
            lambda _: affine(_, 0, [0, .15 * _.size(-2)], 1, [0, 0], self.nearest)  # translate_down
        )

        self.n_operations = args.n_operations
        self.transform = Compose(transform)
        self.data = self.data.numpy()  # (n,h,w)
        self.targets = self.targets.numpy()  # (n,)

    def __getitem__(self, idx):
        _image = Image.fromarray(self.data[idx], mode='L')
        image = self.transform(_image)
        frame = pad(image, (64 - 28) // 2)
        frames = [frame]
        operations = []

        for _ in range(self.n_operations):
            opid = random.choice(range(len(self.operators)))
            operations.append(opid)
            frame = self.operators[opid](frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0).float()
        operations = F.one_hot(torch.from_numpy(np.array(operations, dtype='int64')), len(self.operators)).float()
        
        return frames, operations
