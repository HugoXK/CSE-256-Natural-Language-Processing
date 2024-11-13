# add all  your Encoder and Decoder code here

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5

        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Perform linear operation and split into num_heads
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        # Dropout
        attention = self.dropout(attention)
        
        # Sum weighted values
        x = torch.matmul(attention, value)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        x = self.fc_out(x)
        
        return x, attention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Multi-head attention component
        self.multihead_attention = MultiheadAttention(embed_dim=n_embd, num_heads=n_head)
        # Position-wise feed-forward network
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        # Layer normalization to stabilize the training
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        # Apply layer normalization
        src_processed = self.norm1(src)
        # Change the shape for the multi-head attention
        src_processed = src_processed.permute(1, 0, 2)
        # Apply multi-head self-attention
        attention_output, attention_weights = self.multihead_attention(src_processed, src_processed, src_processed)
        attention_output = attention_output.permute(1, 0, 2)

        # Add and norm step
        src = src + self.dropout(attention_output)
        src_processed = self.norm2(src)
        # Apply the position-wise feed-forward network
        src_processed = self.positionwise_feedforward(src_processed)
        src = src + self.dropout(src_processed)
        return src, attention_weights

def get_embeddings(n_positions, n_embd):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / n_embd) for j in range(n_embd)]
        if pos != 0 else np.zeros(n_embd)
        for pos in range(n_positions)
    ])
    # Apply sinusoidal function to even indices
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    # Apply cosine function to odd indices
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return torch.from_numpy(pos_enc).type(torch.FloatTensor)

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, max_len=512):
        super().__init__()
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Positional embedding layer
        self.positional_embedding = get_embeddings(max_len, n_embd)
        # Stack of Transformer encoder blocks
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(n_embd, n_head) for _ in range(n_layer)])
        # Final layer normalization
        self.final_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Embed tokens and add positional encodings
        x = self.token_embedding(x) + self.positional_embedding[:x.size(1)].unsqueeze(0).to(x.device)
        attention_weights = []
        for layer in self.encoder_layers:
            x, weights = layer(x)
            attention_weights.append(weights)
        # Apply the final normalization
        x = self.final_norm(x)
        return x, attention_weights

class FeedforwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(n_input, n_hidden)
        # ReLU activation function
        self.activation = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # Pass through the first layer
        x = self.fc1(x)
        # Apply activation function
        x = self.activation(x)
        # Pass through the second layer
        x = self.fc2(x)
        return x

class SpeechSegmentModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        # Transformer Encoder
        self.encoder = encoder
        # Feed-forward classifier
        self.classifier = classifier

    def forward(self, src, return_attention=False):
        # Encode input sequence
        encoded, attention_weights = self.encoder(src)
        # Pool the encoded sequence into a single vector per batch
        pooled_output = encoded.mean(dim=1)
        # Classify using the final feed-forward layers
        output = self.classifier(pooled_output)
        if return_attention:
            return output, attention_weights
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Masked multi-head attention component
        self.masked_attention = MultiheadAttention(embed_dim=n_embd, num_heads=n_head)
        # Layer normalization to stabilize the training
        self.norm1 = nn.LayerNorm(n_embd)
        # Position-wise feed-forward network
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        # Second layer normalization
        self.norm2 = nn.LayerNorm(n_embd)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask):
        # Change the shape for the multi-head attention
        x_processed = x.permute(1, 0, 2)
        # Apply masked multi-head self-attention
        attention_output, attention_weights = self.masked_attention(x_processed, x_processed, x_processed, attn_mask=attn_mask)
        attention_output = attention_output.permute(1, 0, 2)
        # Add and norm step
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Apply the position-wise feed-forward network
        feedforward_output = self.positionwise_feedforward(x)
        x = x + self.dropout(feedforward_output)
        x = self.norm2(x)
        return x, attention_weights

    def generate_mask(self, size):
        # Create a future-blinding mask
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, max_len=512):
        super().__init__()
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Positional embedding layer
        self.positional_embedding = get_embeddings(max_len, n_embd)
        # Stack of Transformer decoder blocks
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(n_embd, n_head) for _ in range(n_layer)])
        # Final layer normalization
        self.final_norm = nn.LayerNorm(n_embd)
        # Output projection layer
        self.output_projection = nn.Linear(n_embd, vocab_size)

    def _initialize_parameters(self):
        # Initialize positional embeddings with normal distribution
        nn.init.normal_(self.positional_embedding, mean=0, std=0.02)

    def forward(self, x, return_attention=False):
        attention_weights = []
        seq_len = x.size(1)
        # Generate the future-blinding mask
        attn_mask = self.decoder_layers[0].generate_mask(seq_len).to(x.device)
        # Embed tokens and add positional encodings
        x = self.token_embedding(x) + self.positional_embedding[:seq_len].unsqueeze(0).to(x.device)
        for layer in self.decoder_layers:
            x, weights = layer(x, attn_mask)
            attention_weights.append(weights)

        # Apply the final normalization
        x = self.final_norm(x)
        # Project the outputs to the vocabulary size
        x = self.output_projection(x)
        if return_attention:
            return x, attention_weights
        return x
    
    
## Part 3: Exploration

class TransformerDecoderLayerWithSparseAttention(nn.Module):
    def __init__(self, n_embd, n_head, window_size=4):
        super().__init__()
        self.window_size = window_size
        self.masked_attention = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head)
        self.norm1 = nn.LayerNorm(n_embd)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_processed = x.permute(1, 0, 2)
        B, L, E = x.size()
        
        # Create sparse attention mask
        sparse_mask = torch.full((L, L), float('-inf')).to(x.device)
        for i in range(L):
            for j in range(max(0, i - self.window_size), min(L, i + self.window_size + 1)):
                sparse_mask[i, j] = 0
        
        attention_output, attention_weights = self.masked_attention(x_processed, x_processed, x_processed, attn_mask=sparse_mask)
        attention_output = attention_output.permute(1, 0, 2)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        feedforward_output = self.positionwise_feedforward(x)
        x = x + self.dropout(feedforward_output)
        x = self.norm2(x)
        return x, attention_weights

class TransformerDecoderWithSparseAttention(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, n_embd))
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayerWithSparseAttention(n_embd, n_head) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embd)
        self.output_projection = nn.Linear(n_embd, vocab_size)

    def forward(self, x, return_attention=False):
        attention_weights = []
        x = self.token_embedding(x) + self.positional_embedding[:, :x.size(1), :].to(x.device)
        for layer in self.decoder_layers:
            x, weights = layer(x)
            attention_weights.append(weights)

        x = self.final_norm(x)
        x = self.output_projection(x)
        if return_attention:
            return x, attention_weights
        return x

# import torch.nn.functional as F

# class TransformerDecoderLayerWithDisentangledAttention(nn.Module):
#     def __init__(self, n_embd, n_head):
#         super().__init__()
#         self.n_head = n_head
#         self.n_embd = n_embd
#         self.head_dim = n_embd // n_head
#         assert self.head_dim * n_head == self.n_embd, "Embed_dim must be divisible by n_heads"
        
#         self.scaling = self.head_dim ** -0.5
        
#         # Separate linear layers for queries, keys, and values
#         self.q_proj = nn.Linear(n_embd, n_embd)
#         self.k_proj = nn.Linear(n_embd, n_embd)
#         self.v_proj = nn.Linear(n_embd, n_embd)
        
#         # Output projection layer
#         self.output_proj = nn.Linear(n_embd, n_embd)
        
#         # Layer normalization to stabilize the training
#         self.norm1 = nn.LayerNorm(n_embd)
#         # Position-wise feed-forward network
#         self.positionwise_feedforward = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             nn.ReLU(),
#             nn.Linear(4 * n_embd, n_embd)
#         )
#         # Second layer normalization
#         self.norm2 = nn.LayerNorm(n_embd)
#         # Dropout to prevent overfitting
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, attn_mask):
#         batch_size, seq_len, embed_dim = x.shape
        
#         # Project the queries, keys, and values
#         queries = self.q_proj(x)
#         keys = self.k_proj(x)
#         values = self.v_proj(x)
        
#         # Reshape and permute for multi-head attention processing
#         queries = queries.contiguous().view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         keys = keys.contiguous().view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         values = values.contiguous().view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
#         # Scaled dot-product attention
#         attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scaling
        
#         if attn_mask is not None:
#             attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
#         attn_probs = F.softmax(attn_scores, dim=-1)
        
#         # Apply the attention to the values
#         attn_output = torch.matmul(attn_probs, values)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
#         # Final linear layer to project the concatenated heads
#         attention_output = self.output_proj(attn_output)
        
#         # Add and norm step
#         x = x + self.dropout(attention_output)
#         x = self.norm1(x)

#         # Apply the position-wise feed-forward network
#         feedforward_output = self.positionwise_feedforward(x)
#         x = x + self.dropout(feedforward_output)
#         x = self.norm2(x)

#         return x, attn_probs

#     def generate_mask(self, size):
#         # Create a future-blinding mask
#         mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

# class TransformerDecoderWithDisentangledAttention(nn.Module):
#     def __init__(self, n_layer, n_embd, n_head, vocab_size, max_len=512):
#         super().__init__()
#         # Token embedding layer
#         self.token_embedding = nn.Embedding(vocab_size, n_embd)
#         # Positional embedding layer
#         self.positional_embedding = nn.Parameter(torch.zeros(max_len, n_embd))
#         # Stack of Transformer decoder blocks with disentangled attention
#         self.decoder_layers = nn.ModuleList([
#             TransformerDecoderLayerWithDisentangledAttention(n_embd, n_head) 
#             for _ in range(n_layer)
#         ])
#         # Final layer normalization
#         self.final_norm = nn.LayerNorm(n_embd)
#         # Output projection layer
#         self.output_projection = nn.Linear(n_embd, vocab_size)
#         self._initialize_parameters()

#     def _initialize_parameters(self):
#         # Initialize positional embeddings with normal distribution
#         nn.init.normal_(self.positional_embedding, mean=0, std=0.02)

#     def forward(self, x, return_attention=False):
#         attention_weights = []
#         seq_len = x.size(1)
        
#         # Generate the future-blinding mask
#         attn_mask = self.decoder_layers[0].generate_mask(seq_len).to(x.device)
        
#         # Embed tokens and add positional encodings
#         x = self.token_embedding(x) + self.positional_embedding[:seq_len, :].unsqueeze(0).to(x.device)
        
#         for layer in self.decoder_layers:
#             x, weights = layer(x, attn_mask)
#             attention_weights.append(weights)

#         # Apply the final normalization
#         x = self.final_norm(x)
#         # Project the outputs to the vocabulary size
#         x = self.output_projection(x)
        
#         if return_attention:
#             return x, attention_weights
#         return x