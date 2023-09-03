from encoder import EncoderLayer
from decoder import DecoderLayer
from embedder import InputEmbedderAndEncoder, OutputEmbedderAndEncoder
import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embedding_size: int = 512, num_heads: int = 8,
                 num_layers: int = 4, d_hidden: int = 2048, max_seq_len: int = 50, dropout: float = 0.01):
        super(Transformer, self).__init__()
        self.encoder_embedding = InputEmbedderAndEncoder(src_vocab_size, max_seq_len=max_seq_len, embedding_size=embedding_size)
        self.decoder_embedding = OutputEmbedderAndEncoder(tgt_vocab_size, max_seq_len=max_seq_len, embedding_size=embedding_size)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_size, num_heads, d_hidden, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_size, num_heads, d_hidden, dropout) for _ in range(num_layers)])

        self.linear_layer = nn.Linear(embedding_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    @typechecked
    def forward(self, tokenized_src_sentences: TensorType['batch_size', 'tokens_num_src', int],
                tokenized_tgt_sentences: TensorType['batch_size', 'tokens_num_tgt', int]):
        src_embedded, src_mask = self.encoder_embedding(tokenized_src_sentences)
        tgt_embedded, tgt_mask = self.decoder_embedding(tokenized_tgt_sentences)
        
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        enc_out = src_embedded
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out, src_mask)

        dec_out = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out, enc_out, src_mask, tgt_mask)

        out = self.linear_layer(dec_out)
        return out
