import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_size: int, d_hidden: int):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(embedding_size, d_hidden)
        self.linear_layer_2 = nn.Linear(d_hidden, embedding_size)
        self.relu = nn.ReLU()

    @typechecked
    def forward(self, x: TensorType['batch_size', 'num_of_words', 'embedding_size', float]) -> TensorType['batch_size', 'num_of_words', 'embedding_size', float]:
        x = self.linear_layer_1(x)
        x = self.relu(x)
        x = self.linear_layer_2(x)
        return x

