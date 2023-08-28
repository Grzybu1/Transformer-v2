import torch
import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class MultiHeadAttention(nn.Module):
    """Some Information about MultiHeadedAttention"""
    def __init__(self, embedding_size: int = 512, heads_num: int = 8, value_dimension: int = 6):
        super(MultiHeadAttention, self).__init__()
        
        self.embedding_size = embedding_size
        self.heads_num = heads_num
        
        assert self.embedding_size % self.heads_num == 0
        self.k_dimension = int(embedding_size/heads_num)
        
        self.value_dimension = value_dimension
        query_transform_matrices = nn.ModuleList([nn.Linear(self.embedding_size, self.k_dimension) for _ in range(self.heads_num)])
        key_transform_matrices = nn.ModuleList([nn.Linear(self.embedding_size, self.k_dimension) for _ in range(self.heads_num)])
        value_transform_matrices = nn.ModuleList([nn.Linear(self.embedding_size, self.value_dimension) for _ in range(self.heads_num)])
        self.heads = [{'query_matrix': qm, 'key_matrix': km, 'value_matrix': vm} for qm, km, vm in zip(query_transform_matrices, key_transform_matrices, value_transform_matrices)]
        self.normalization_matrix = nn.Linear(self.heads_num * self.value_dimension, self.embedding_size)
    
    @typechecked
    def MaskedAttention(self, Q:TensorType['batch_size', 'num_of_words', 'k_dimension', float],
                K:TensorType['batch_size', 'num_of_words', 'k_dimension', float],
                V:TensorType['batch_size', 'num_of_words', 'v_dimension', float],
                mask: TensorType['batch_size', 'num_of_it_for_single_sequence', 'num_of_words', bool]) -> TensorType['batch_size', 'num_of_words', 'v_dimension', float] :
        
        batch_size = Q.shape[0]
        num_of_words = Q.shape[1]
        dot_product = torch.matmul(Q, K.transpose(-2,-1)) / (self.embedding_size**-0.5)
        assert list(dot_product.shape) == [batch_size, num_of_words, num_of_words]
        
        dot_product = dot_product.masked_fill(mask == False, -1e9)
        value_weights = torch.softmax(dot_product, dim=-1)
        new_values = torch.matmul(value_weights, V)
        return new_values
        
    
    @typechecked
    def forward(self, Q:TensorType['batch_size', 'num_of_words', 'embedding_size', float],
                K:TensorType['batch_size', 'num_of_words', 'embedding_size', float],
                V:TensorType['batch_size', 'num_of_words', 'embedding_size', float],
                mask: TensorType['batch_size', 'num_of_it_for_single_sequence', 'num_of_words', bool]) -> TensorType['batch_size', 'num_of_words', 'embedding_size', float]:
        batch_size = Q.shape[0]
        num_of_words = Q.shape[1]
        value_results = []
        for head in self.heads:
            query_matrix = head['query_matrix']
            key_matrix = head['key_matrix']
            value_matrix = head['value_matrix']
            Q_i = query_matrix(Q)
            assert list(Q_i.shape) == [batch_size, num_of_words, self.k_dimension]
            K_i = key_matrix(K)
            assert list(K_i.shape) == [batch_size, num_of_words, self.k_dimension]
            V_i = value_matrix(V)
            assert list(V_i.shape) == [batch_size, num_of_words, self.value_dimension]
            
            value_results.append(self.MaskedAttention(Q_i, K_i, V_i, mask))
            
        concatenated_attention_results = torch.cat(value_results, dim=-1)
        assert list(concatenated_attention_results.shape) == [batch_size, num_of_words, self.heads_num * self.value_dimension]
        
        multihead_attention_result = self.normalization_matrix(concatenated_attention_results)
        return multihead_attention_result