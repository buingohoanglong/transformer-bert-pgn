import torch
from Dictionary import Dictionary, preprocess
from torch.nn.utils.rnn import pad_sequence


def process_batch(src_batch, tgt_batch, dictionary, tokenizer, segmenter, max_src_len=256, use_pgn=False):
    """
    input: list [batch_size] of tensors in different lengths
    output: tensor [batch_size, seq_length]
    """
    assert len(src_batch) == len(tgt_batch)
    dictionary_ext = None
    if use_pgn:
        dictionary_ext = Dictionary(tokenizer=tokenizer)
        dictionary_ext.token2index = {**dictionary.token2index}
        dictionary_ext.index2token = {**dictionary.index2token}
        dictionary_ext.vocab_size = dictionary.vocab_size
    src = []
    tgt = []
    src_bert = []
    src_ext = [] if use_pgn else None
    tgt_ext = [] if use_pgn else None
    for idx in range(len(src_batch)):
        src_str = preprocess(segmenter, src_batch[idx].strip())
        tgt_str = preprocess(segmenter, tgt_batch[idx].strip())
        src.append(dictionary.encode(src_str, append_bos=False, append_eos=True))
        tgt.append(dictionary.encode(tgt_str, append_bos=True, append_eos=True))
        src_bert.append(torch.tensor(tokenizer.encode(src_str)[1:]))
        if use_pgn:
            src_ext.append(dictionary_ext.encode(src_str, append_bos=False, append_eos=True, update=True))
            tgt_ext.append(dictionary_ext.encode(tgt_str, append_bos=True, append_eos=True, update=True))

    src = pad_sequence(src, padding_value=dictionary.token_to_index(dictionary.pad_token), batch_first=True)
    tgt = pad_sequence(tgt, padding_value=dictionary.token_to_index(dictionary.pad_token), batch_first=True)
    src_bert = pad_sequence(src_bert, padding_value=tokenizer.pad_token_id, batch_first=True)
    if use_pgn:
        src_ext = pad_sequence(src_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token), batch_first=True)
        tgt_ext = pad_sequence(tgt_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token), batch_first=True)
    assert src.size(1) == src_bert.size(1)
    # Truncate if seq_len exceed max_src_length
    if src.size(1) > max_src_len:
        src = src[:,:max_src_len]
        src_bert = src_bert[:,:max_src_len]
        if use_pgn:
            src_ext = src_ext[:,:max_src_len]
    return src, tgt, src_bert, src_ext, tgt_ext, dictionary_ext