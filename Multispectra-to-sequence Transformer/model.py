"""
Transformer for spectra to sequence reconstruction
"""

import torch
import torch.nn as nn
import math

def init_weights(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (B, L) --> (B, L, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class ContinuousEmbeddings(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.continuous_embedding = nn.Linear(1, d_model)
    
    def forward(self, x):
        # (B, spec_length) --> (B, spec_length, 1) --> (B, spec_length, d_model)
        x = x.unsqueeze(-1)
        return self.continuous_embedding(x) * math.sqrt(self.d_model)    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, period=1e4):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.period = period

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create tensor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(self.period) / d_model))

        # apply sin to even positions, cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # expand to batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        return x
    
class LayerNormalization(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # added

    def forward(self, x):
        # mean and std across each model dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff) # W1, b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2, b2

    def forward(self, x):
        # (B, L, d_model) --> (B, L, d_ff) --> (B, L, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (B, h, L, d_k) . (B, h, d_k, L) = (B, h, L, L)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (B, h, L, L) . (B, h, L, d_k) = (B, h, L, d_k)
        #return (attention_scores @ value), attention_scores
        return (attention_scores @ value)
    
    def forward(self, q, k, v, mask):
        # (B, L, d_model) --> (B, L, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (B, L, d_model) --> (B, L, h, d_k) --> (B, h, L, d_k) ==> each head sees (L, d_k)
        query = query.view(-1, query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(-1, key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(-1, value.shape[1], self.h, self.d_k).transpose(1, 2)

        #x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (B, h, L, d_k) --> (B, L, h, d_k) --> (B, L, d_model) 
        x = x.transpose(1, 2).contiguous().view(-1, q.shape[1], self.d_model)

        # (B, L, d_model) --> (B, L, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer, norm_first=True):
        if norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block:MultiHeadAttentionBlock, 
                 feed_forward_block:FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # note: src mask for padding
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, d_model, encoder_layers:nn.ModuleList):
        super().__init__()
        self.layers = encoder_layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block:MultiHeadAttentionBlock,
                 cross_attention_block:MultiHeadAttentionBlock,
                 feed_forward_block:FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, encoder_output, src_mask, x, tgt_mask):
        # note: tgt_mask to avoid looking ahead
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, d_model, decoder_layers:nn.ModuleList):
        super().__init__()
        self.layers = decoder_layers
        self.norm = LayerNormalization(d_model)

    def forward(self, encoder_output, src_mask, x, tgt_mask):
        for layer in self.layers:
            x = layer(encoder_output, src_mask, x, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (B, L, d_model) --> (B, L, d_vocab)
        return self.projection(x)
    
class Transformer(nn.Module):
    """
    Wrapper class for encoding spectra and decoding sequences using
    transformers.
    """
    def __init__(self, spectrum_embedder, spectrum_pos_enc, seq_embedder,
                 seq_pos_enc, encoder, decoder, projection_layer):
        
        super().__init__()  
        self.spectrum_embedder = spectrum_embedder
        self.seq_embedder = seq_embedder

        self.spectrum_pos_enc = spectrum_pos_enc
        self.seq_pos_enc = seq_pos_enc
        
        self.encoder = encoder
        self.decoder = decoder

        self.projection_layer = projection_layer

        self.apply(init_weights)
        
    def encode(self, spectra):
        spectra = self.spectrum_embedder(spectra)
        spectra = self.spectrum_pos_enc(spectra)
        return self.encoder(spectra, src_mask=None)

    def decode(self, encoded_spectra, seqs, seq_mask):
        seqs = self.seq_embedder(seqs)
        seqs = self.seq_pos_enc(seqs)
        return self.decoder(encoded_spectra, None, seqs, seq_mask)
    
    def project(self, seqs):
        return self.projection_layer(seqs)
    
    def forward(self, spectra, seqs, seq_mask=None):
        """
        Tensor sizes
        spectra : (B, spec_length)
        seqs : (B, L)
        seq_mask : (L, L)
        returns : (B, L, vocab_size)
        """
        # feed through encoder and decoder
        encoded_spectra = self.encode(spectra)
        decoded_seqs = self.decode(encoded_spectra, seqs, seq_mask)

        # project to vocab size
        projected_seqs = self.project(decoded_seqs)

        return projected_seqs
    
def build_transformer(spec_length, seq_length, vocab_size, d_model=128, h=8, 
                      N=6, d_ff=2048, dropout=0.1, seq_embedder=None):
    """
    Function to build a transformer model for spectra to sequence tasks.
    """
    spectrum_embedder = ContinuousEmbeddings(d_model)
    if seq_embedder is None:
        seq_embedder = InputEmbeddings(d_model, vocab_size)

    spectrum_pos_enc = PositionalEncoding(d_model, spec_length)
    seq_pos_enc = PositionalEncoding(d_model, seq_length)

    # build encoder/decoder layers
    encoder_layers = []
    decoder_layers = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_layers.append(encoder_block)

        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_layers.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_layers))
    decoder = Decoder(d_model, nn.ModuleList(decoder_layers))
    
    projection_layer = ProjectionLayer(d_model, vocab_size)

    transformer = Transformer(spectrum_embedder, spectrum_pos_enc, 
                              seq_embedder, seq_pos_enc, encoder, decoder,
                              projection_layer)
    
    return transformer

class MultiSpectraTransformer(nn.Module):
    """
    Wrapper class for encoding multiple spectra and decoding sequences using
    transformers.
    """
    def __init__(self, spectrum_transformers, projection_layer=None):
        
        super().__init__()
        self.spectrum_transformers = spectrum_transformers
        self.projection_layer = projection_layer
        
    def forwardTransformers(self, spectra, seqs, seq_mask):
        # encode each spectrum type and decode sequences
        decoded_tokens = []
        for spec_type in spectra:
            spectrum = spectra[spec_type]
            transformer = self.spectrum_transformers[spec_type]
            decoded_token = transformer(spectrum, seqs, seq_mask)
            decoded_tokens.append(decoded_token)

        # concatenate along model dimension
        decoded_tokens = torch.cat(decoded_tokens, dim=-1)
        return decoded_tokens
    
    def project(self, seqs):
        # (B, L, d_vocab * num_spectra) --> (B, L, vocab_size)
        return self.projection_layer(seqs)
    
    def forward(self, spectra, seqs, seq_mask=None):
        """
        Tensor sizes
        spectra : dict of (B, spec_length)
        returns : (B, L, vocab_size)
        """
        # feed through each transformer
        decoded_tokens = self.forwardTransformers(spectra, seqs, seq_mask)

        # probability reweighting
        if self.projection_layer is None:
            return decoded_tokens
        
        projected_seqs = self.project(decoded_tokens)

        return projected_seqs
    
def build_multi_spectra_transformer(spec_lengths, seq_length, vocab_size, d_model=128, h=8, 
                                    N=6, d_ff=2048, dropout=0.1):
    """
    Function to build a multi-spectra transformer model for spectra to sequence tasks.
    """
    spectrum_transformers = nn.ModuleDict()

    # shared sequence embedder
    seq_embedder = InputEmbeddings(d_model, vocab_size)

    for spec_type in spec_lengths:
        spec_length = spec_lengths[spec_type]
        spectrum_transformer = build_transformer(spec_length, seq_length, vocab_size, d_model, h, N, d_ff, dropout, seq_embedder)
        spectrum_transformers[spec_type] = spectrum_transformer

    if len(spectrum_transformers) == 1:
        projection_layer = None
    else:
        projection_layer = ProjectionLayer(vocab_size * len(spec_lengths), vocab_size)

    multi_spectra_transformer = MultiSpectraTransformer(spectrum_transformers, projection_layer)
    
    return multi_spectra_transformer