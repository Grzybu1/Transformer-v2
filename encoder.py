from pw_feed_forward import PositionWiseFeedForward
from multiheaded_attention import MultiHeadAttention
import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, d_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_size, num_heads)
        self.pw_feed_forward = PositionWiseFeedForward(embedding_size, d_hidden)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
    
    @typechecked
    def forward(self, x: TensorType['batch_size', 'num_of_words', 'embedding_size', float],
                mask: TensorType['batch_size', 'num_of_it_for_single_sequence', 'max_seq_len', bool]) -> TensorType['batch_size', 'num_of_words', 'embedding_size', float]:
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        x = self.pw_feed_forward(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        return x