import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, dropout, bert, d_bert, padding_idx=None, use_pgn=False, unk_idx=None):
        super(Transformer, self).__init__()
        self.src_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, dropout=dropout)
        self.tgt_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, dropout=dropout)
        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_bert=d_bert)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_bert=d_bert)
        self.linear = nn.Linear(d_model, vocab_size)
        self.bert = bert
        self.padding_idx = padding_idx
        self.use_pgn = use_pgn
        self.unk_idx = unk_idx
        self.pointer_generator = PointerGeneratorNetwork(d_model=d_model, dropout=dropout, vocab_size=vocab_size)

    def forward(self, src, tgt, src_bert, src_ext, max_oov_len):
        """
        :param mandatory Tensor[batch_size, src_seq_len] src: index of source tokens
        :param mandatory Tensor[batch_size, tgt_seq_len] tgt: index of target tokens
        :param optional Tensor[batch_size, src_seq_len] src_bert: index of source bert tokens
        """
        if self.use_pgn:
            assert (src_ext is not None) and (max_oov_len is not None)
            assert src_ext.size() == src.size()
            assert max_oov_len >= 0

        src_padding_mask = src.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
        tgt_padding_mask = tgt.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
        special_token_mask = src.eq(self.unk_idx).unsqueeze(1) if self.unk_idx is not None else None

        src_embedding = self.src_embedding(src)
        bert_embedding = self.bert(src_bert).last_hidden_state.detach()
        encoder_out = self.encoder(src_embedding, bert_embedding, padding_mask=src_padding_mask)

        tgt_embedding = self.tgt_embedding(tgt)
        decoder_out = self.decoder(
            tgt_embedding, encoder_out, bert_embedding, 
            src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask
        )

        final_dist = self.pointer_generator(
            encoder_out=encoder_out, 
            decoder_out=decoder_out, 
            decoder_in=tgt_embedding, 
            src_ext=src_ext, 
            max_oov_len=max_oov_len, 
            padding_mask=src_padding_mask, 
            special_token_mask=special_token_mask
        ) if self.use_pgn else self.linear(decoder_out) 

        return final_dist


class PointerGeneratorNetwork(nn.Module):
    def __init__(self, d_model, dropout, vocab_size):
        super(PointerGeneratorNetwork, self).__init__()
        self.encoder_decoder_attention = Attention(d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_model, d_v=d_model, dropout=dropout)
        self.vocab_projection = nn.Linear(d_model, vocab_size)
        self.context_to_pgen = nn.Linear(d_model, 1)
        self.decoder_out_to_pgen = nn.Linear(d_model, 1)
        self.decoder_in_to_pgen = nn.Linear(d_model, 1)

    def forward(self, encoder_out, decoder_out, decoder_in, src_ext, max_oov_len, padding_mask=None, special_token_mask=None):
        """
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] decoder_out
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] decoder_in
        :param mandatory Tensor[batch_size, src_seq_len] src_ext: ids of tokens in extend dictionary
        :param optional Tensor[batch_size, 1, src_seq_len] padding_mask: prevent attention with padding token in src
        :param optional Tensor[batch_size, 1, src_seq_len] special_token_mask: prevent attention with special tokens in src
        """
        context, att_dist = self.encoder_decoder_attention(
            decoder_out, encoder_out, encoder_out, padding_mask=padding_mask, special_token_mask=special_token_mask
        ) # [batch_size, tgt_seq_len, d_model], [batch_size, tgt_seq_len, src_seq_len]

        vocab_dist = F.softmax(self.vocab_projection(decoder_out), dim=-1) # [batch_size, tgt_seq_len, vocab_size]
        extra_zeros = torch.zeros(vocab_dist.size(0), vocab_dist.size(1), max_oov_len).to(vocab_dist.device) # [batch_size, tgt_seq_len, max_oov_len]
        extend_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1) # [batch_size, tgt_seq_len, extend_vocab_size], extend_vocab_size = vocab_size + max_oov_len

        p_gen = torch.sigmoid(
            self.context_to_pgen(context) + self.decoder_out_to_pgen(decoder_out) + self.decoder_in_to_pgen(decoder_in)
        ) # [batch_size, tgt_seq_len, 1]

        # combine distribution
        extend_vocab_dist = p_gen * extend_vocab_dist # [batch_size, tgt_seq_len, extend_vocab_size]
        att_dist = (1.0 - p_gen) * att_dist # [batch_size, tgt_seq_len, src_seq_len]

        final_dist = extend_vocab_dist.scatter_add(
            dim=-1,
            index=src_ext.unsqueeze(1).repeat(1, extend_vocab_dist.size(1), 1), # [batch_size, tgt_seq_len,  src_seq_len]
            src=att_dist
        ) # [batch_size, tgt_seq_len, extend_vocab_size]

        return final_dist


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout, d_bert):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout, d_bert=d_bert)
            for _ in range(num_layers)
        ])

    def forward(self, x, bert_embedding, padding_mask=None):
        for layer in self.encoder_layers:
            x = layer(x, bert_embedding, padding_mask=padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout, d_bert):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout, d_bert=d_bert)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_out, bert_embedding, src_padding_mask=None, tgt_padding_mask=None):
        """
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] x: embedding of previous decoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out: embedding of final encoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_bert] bert_embedding: embedding of bert
        :param optional Tensor[batch_size, 1, src_seq_len] src_padding_mask: padding mask for key from decoder
        :param optional Tensor[batch_size, 1, tgt_seq_len] tgt_padding_mask: padding mask for key from encoder, bert
        """
        for layer in self.decoder_layers:
            x = layer(
                x, encoder_out, bert_embedding, 
                src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask
            )
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, d_bert):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(EncoderLayer, self).__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        assert d_k * num_heads == d_model and d_v * num_heads == d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.bert_enc_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_bert, d_V_in=d_bert, d_k=d_k, d_v=d_v, dropout=dropout)
        self.attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.feed_forward_residual = Residual(d_model=d_model,dropout=dropout)


    def forward(self, x, bert_embedding, padding_mask=None):
        """
        :param Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        self_attention = self.self_attention(x, x, x, padding_mask=padding_mask)
        bert_enc_attention = self.bert_enc_attention(x, bert_embedding, bert_embedding, padding_mask=padding_mask)
        ratios = self.get_ratios()
        bertfused_attention = self.attention_residual(
            ratios[0]*self_attention + ratios[1]*bert_enc_attention,
            x
        )

        return self.feed_forward_residual(
            self.feed_forward(bertfused_attention), 
            bertfused_attention
        )

    def get_ratios(self):
        return [0.5, 0.5]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, d_bert):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(DecoderLayer, self).__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        assert d_k * num_heads == d_model and d_v * num_heads == d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout, masking=True)
        self.self_attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.encoder_decoder_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout, masking=True)
        self.bert_dec_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_bert, d_V_in=d_bert, d_k=d_k, d_v=d_v, dropout=dropout, masking=True) 
        self.bertfused_attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.feed_forward_residual = Residual(d_model=d_model,dropout=dropout)

    def forward(self, x, encoder_out, bert_embedding, src_padding_mask=None, tgt_padding_mask=None):
        """
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] x: embedding of previous decoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out: embedding of final encoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_bert] bert_embedding: embedding of bert
        :param optional Tensor[batch_size, 1, src_seq_len] src_padding_mask: padding mask for key from decoder
        :param optional Tensor[batch_size, 1, tgt_seq_len] tgt_padding_mask: padding mask for key from encoder, bert
        """
        self_attention = self.self_attention_residual(
            self.self_attention(x, x, x, padding_mask=tgt_padding_mask), 
            x
        )

        enc_dec_attention = self.encoder_decoder_attention(
            self_attention, encoder_out, encoder_out, padding_mask=src_padding_mask
        )
        bert_dec_attention = self.bert_dec_attention(
            self_attention, bert_embedding, bert_embedding, padding_mask=src_padding_mask
        )
        ratios = self.get_ratios()
        bertfused_attention = self.bertfused_attention_residual(
            ratios[0]*enc_dec_attention + ratios[1]*bert_dec_attention,
            self_attention
        )
        return self.feed_forward_residual(
            self.feed_forward(bertfused_attention),
            bertfused_attention
        )
    
    def get_ratios(self):
        return [0.5, 0.5]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_Q_in, d_K_in, d_V_in, d_k, d_v, dropout, masking=False):
        """
        :param mandatory int num_heads
        :param mandatory int d_Q_in
        :param mandatory int d_K_in
        :param mandatory int d_V_in
        :param mandatory int d_k
        :param mandatory int d_v
        """
        super(MultiHeadAttention, self).__init__()
        self.masking = masking
        self.heads = nn.ModuleList([
        Attention(d_Q_in=d_Q_in, d_K_in=d_K_in, d_V_in=d_V_in, d_k=d_k, d_v=d_v, dropout=dropout) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * d_v, d_Q_in)

    def forward(self, Q, K, V, padding_mask=None):
        """
        :param mandatory Tensor[batch_size, Q_seq_length, d_Q_in] Q: query
        :param mandatory Tensor[batch_size, K_seq_length, d_K_in] K: key
        :param mandatory Tensor[batch_size, K_seq_length, d_V_in] V: value
        :return Tensor[batch_size, Q_seq_length, d_v]
        """
        attention_mask = None
        if self.masking:
            attention_mask = torch.ones(Q.size(0), Q.size(1), K.size(1)).triu(diagonal=1).type(torch.bool).to(Q.device) # [batch_size, Q_seq_length, K_seq_length]
        concat_heads = torch.cat([
            h(Q, K, V, attention_mask=attention_mask, padding_mask=padding_mask)[0]
            for h in self.heads
        ], dim=-1) # [batch_size, Q_seq_length, num_heads * d_v]
        return self.linear(concat_heads) # [batch_size, Q_seq_length, d_model]


class Attention(nn.Module):
    def __init__(self, d_Q_in, d_K_in, d_V_in, d_k, d_v, dropout):
        """
        :param mandatory int d_Q_in
        :param mandatory int d_K_in
        :param mandatory int d_V_in
        :param mandatory int d_k
        :param mandatory int d_v
        """
        super(Attention, self).__init__()
        self.q = nn.Linear(d_Q_in, d_k)
        self.k = nn.Linear(d_K_in, d_k)
        self.v = nn.Linear(d_V_in, d_v)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_v, d_v)

    def forward(self, Q, K, V, attention_mask=None, padding_mask=None, special_token_mask=None):
        """
        :param mandatory Tensor[batch_size, Q_seq_length, d_Q_in] Q: query
        :param mandatory Tensor[batch_size, K_seq_length, d_K_in] K: key
        :param mandatory Tensor[batch_size, K_seq_length, d_V_in] V: value
        :param optional Tensor[batch_size, 1, K_seq_length] padding_mask: masking for padding
        :param optional Tensor[batch_size, Q_seq_length, K_seq_length] attention_mask: masking for attention
        :param optional Tensor[batch_size, 1, K_seq_length] special_token_mask: masking for special tokens (oov_token, number, named entity,...) -> used in PGN
        :return Tensor[batch_size, Q_seq_length, d_v] context_matrix, Tensor[batch_size, Q_seq_length, K_seq_length] attention_distribution
        """
        query = self.q(Q) # [batch_size, Q_seq_length, d_k]
        key = self.k(K) # [batch_size, K_seq_length, d_k]
        value = self.v(V) # [batch_size, K_seq_length, d_v]
        scale = query.size(-1) ** 0.5 # scalar
        attention_weight = query.bmm(key.transpose(1,2)) / scale # [batch_size, Q_seq_length, K_seq_length]
        if attention_mask is not None:
            attention_weight.masked_fill_(attention_mask, float("-Inf")) # repalce masked position by -inf
        if padding_mask is not None:
            # do not attend to padding symbols
            attention_weight.masked_fill_(padding_mask, float("-Inf")) # repalce masked position by -inf
        if special_token_mask is not None:
            attention_weight.masked_fill_(special_token_mask, float("-Inf")) # repalce masked position by -inf
        score = F.softmax(attention_weight, dim=-1) # [batch_size, Q_seq_length, K_seq_length]
        score = self.dropout(score)
        return self.linear(score.bmm(value)), score # [batch_size, Q_seq_length, d_v], [batch_size, Q_seq_length, K_seq_length]


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout):
        """
        :param madatory int vocab_size
        :param madatory int d_model
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_length] x: each sequence is a "list" of vocab index [0, vocab_size)
        :return Tensor[batch_size, seq_length, d_model]
        """
        return self.dropout(self.embedding(x)*self.embedding_scale + self.positional_encoding(x))



class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        """
        :param madatory int d_model
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model]
        """
        positions = torch.arange(x.size(1)).unsqueeze(1).to(x.device) # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(x.device) # [1, d_model]
        angles = self.get_angles(positions, indexes) # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2]) # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2]) # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(x.size(0), 1, 1) # [batch_size, seq_len, d_model]
        return position_encoding



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(FeedForward, self).__init__()
        self.first_linear = nn.Linear(d_model, d_ff)
        self.second_linear = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model]
        """
        x = self.first_linear(x)
        x = F.relu(x)
        return self.second_linear(x)

class Residual(nn.Module):
    def __init__(self, d_model, dropout):
        """
        :param mandatory nn.Module sublayer
        :param mandatory int d_model
        """
        super(Residual, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_non_dropout):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        return self.layer_norm(self.dropout(x) + x_non_dropout)
