from transformers import BartTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC

import torch
import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class EmbedderAndEncoderBase(nn.Module, ABC):
    """Base class for embedding and positional encoding"""
    
    def __init__(self, dict_size: int, embedding_size: int = 512, max_seq_len: int = 50, n_parameter: int = 10000):
        super(EmbedderAndEncoderBase, self).__init__()
        self.max_seq_len=max_seq_len
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_size)
        self.generate_positional_encodings(max_seq_len, n_parameter)
        
    @typechecked
    def generate_positional_encodings(self, max_seq_len: int, n_parameter: int) -> None:
        np_positional_encodings_matrix = np.zeros([max_seq_len, self.embedding_size])
        for sentence_pos in np.arange(max_seq_len):
            for dimension_pos in np.arange(self.embedding_size//2):
                theta = sentence_pos / (n_parameter ** (2*dimension_pos/self.embedding_size))
                np_positional_encodings_matrix[sentence_pos, 2*dimension_pos] = np.sin(theta)
                np_positional_encodings_matrix[sentence_pos, 2*dimension_pos+1] = np.cos(theta)
        positional_encodings_matrix = torch.tensor(data=np_positional_encodings_matrix, dtype=float)
        self.register_buffer("positional_encodings_matrix", positional_encodings_matrix)
                
    @typechecked
    def forward(self, tokenized_sentences: TensorType['batch_size', 'tokens_num', int]) -> (TensorType['batch_size', 'tokens_num', 'embedding_size', float], TensorType['batch_size', 'max_seq_len', bool]):
        embeddings = self.embedding(tokenized_sentences)
        with torch.no_grad():
            positional_encoding = self.positional_encodings_matrix[:tokenized_sentences.shape[1]]
            encoded_embeddings = embeddings * np.sqrt(self.embedding_size) + positional_encoding
            mask = self.generate_mask(tokenized_sentences)
        return encoded_embeddings.to(torch.float32), mask

class InputEmbedderAndEncoder(EmbedderAndEncoderBase):
    def __init__(self, dict_size: int, embedding_size: int = 512, max_seq_len: int = 50, n_parameter: int = 10000):
        super(InputEmbedderAndEncoder, self).__init__(dict_size, embedding_size, max_seq_len, n_parameter)
        
    @typechecked
    def generate_mask(self, tokenized_sentences: TensorType['batch_size', 'tokens_num', int], pad_token_id: int = 1) -> TensorType['batch_size', 'num_of_it_for_single_sequence', 'max_seq_len', bool]:
        mask = tokenized_sentences != pad_token_id
        return mask.unsqueeze(1)

class OutputEmbedderAndEncoder(EmbedderAndEncoderBase):
    def __init__(self, dict_size: int, embedding_size: int = 512, max_seq_len: int = 50, n_parameter: int = 10000):
        super(OutputEmbedderAndEncoder, self).__init__(dict_size, embedding_size, max_seq_len, n_parameter)
    
    @typechecked
    def generate_mask(self, tokenized_sentences: TensorType['batch_size', 'tokens_num', int], pad_token_id: int = 1) -> TensorType['batch_size', 'num_of_it_for_single_sequence', 'max_seq_len', bool]:
        mask = (tokenized_sentences != pad_token_id).unsqueeze(2)
        sentence_len = tokenized_sentences.shape[1]
        nopeak = (torch.tril(torch.ones(1, sentence_len, sentence_len), diagonal=0)).bool()
        mask = mask & nopeak
        return mask

if __name__ == "__main__":
    max_seq_len = 6
    embedding_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    model = BartTokenizer.from_pretrained("facebook/bart-base")
    # print(model.pad_token_id)
    sentences = ['I am',
                # 'I feel well.',
                # 'I\'m feeling great',
                # 'I am feeling great',
                'I feel very.']
    sentences_tokens = []
    sentences_tokens = torch.tensor(data=model(sentences, padding='max_length', max_length=max_seq_len)["input_ids"], dtype=int)
    embedder = InputEmbedderAndEncoder(model.vocab_size, max_seq_len=max_seq_len, embedding_size=embedding_size)
    positional_encodings, mask = embedder.forward(sentences_tokens)
    
    embedder2 = OutputEmbedderAndEncoder(model.vocab_size, max_seq_len=max_seq_len, embedding_size=embedding_size)
    positional_encodings2, mask2 = embedder2.forward(sentences_tokens)
    
    print(positional_encodings.shape)
    print(positional_encodings2.shape)
    print(mask)
    print(mask2)
    positional_encodings_matrix = embedder.positional_encodings_matrix.numpy()
    sns.heatmap(embedder.positional_encodings_matrix.numpy(), cbar_kws={'label': 'Wartość'})
    plt.ylabel("Indeks słowa")
    plt.xlabel('Indeks wektora pozycyjnego')
    # plt.show()
    
    fig, ax = plt.subplots(5, sharex=True, sharey=True)
    for i, word_pos in enumerate(embedder.positional_encodings_matrix.numpy().transpose()[:5]):
        ax[i].plot(word_pos, linestyle='dashed', marker='.', linewidth=0.5)
        ax[i].title.set_text(f'Embedding nr {i}')
    fig.supxlabel('Numer słowa w sekwencji')
    fig.supylabel('Wartości')
    # plt.show()

    # print(words_embeddings)
    # for i, embedding1 in enumerate(words_embeddings):
    #     for j, embedding2 in enumerate(words_embeddings):
    #         np_words_similarity_matrix[i,j] = util.cos_sim(embedding1, embedding2)
            
    # sns.heatmap(np_words_similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=words, yticklabels=words, cbar=False)
    # plt.show()