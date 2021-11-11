import torch
from Dictionary import Dictionary, preprocess
from torch.nn.utils.rnn import pad_sequence


def process_batch(batch, dictionary, tokenizer, annotator, 
                max_src_len=256, use_pgn=False, use_ner=False, device='cpu'):
    """
    input: list [batch_size] of tensors in different lengths
    output: tensor [batch_size, seq_length]
    """
    src_batch, tgt_batch = batch
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
    src_ne = [] if use_ner else None
    for idx in range(len(src_batch)):
        src_preprocessed = preprocess(annotator, src_batch[idx].strip(), ner=use_ner)
        tgt_preprocessed = preprocess(annotator, tgt_batch[idx].strip(), ner=use_ner)
        src_str = " ".join(src_preprocessed['words'])
        tgt_str = " ".join(tgt_preprocessed['words'])
        src_encode = dictionary.encode(src_str, append_bos=False, append_eos=True)
        tgt_encode = dictionary.encode(tgt_str, append_bos=True, append_eos=True)
        src.append(torch.tensor(src_encode['ids']))
        tgt.append(torch.tensor(tgt_encode['ids']))
        src_bert.append(torch.tensor(tokenizer.encode(src_str)[1:]))
        if use_pgn:
            src_ext.append(torch.tensor(
                dictionary_ext.encode(src_str, append_bos=False, append_eos=True, update=True)['ids']
            ))
            tgt_ext.append(torch.tensor(
                dictionary_ext.encode(tgt_str, append_bos=True, append_eos=True, update=True)['ids']
            ))
        if use_ner:
            src_ne.append(
                torch.tensor(ner_for_bpe(
                    bpe_tokens=src_encode['bpe_tokens'], ne_tokens=src_preprocessed['name_entities'], 
                    get_mask=True, special_tokens=[dictionary.bos_token, dictionary.eos_token]
                ))
            )

    src = pad_sequence(src, padding_value=dictionary.token_to_index(dictionary.pad_token), batch_first=True)
    tgt = pad_sequence(tgt, padding_value=dictionary.token_to_index(dictionary.pad_token), batch_first=True)
    src_bert = pad_sequence(src_bert, padding_value=tokenizer.pad_token_id, batch_first=True)
    if use_pgn:
        src_ext = pad_sequence(src_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token), batch_first=True)
        tgt_ext = pad_sequence(tgt_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token), batch_first=True)
    if use_ner:
        src_ne = pad_sequence(src_ne, padding_value=0, batch_first=True)
    assert src.size(1) == src_bert.size(1)
    # Truncate if seq_len exceed max_src_length
    if src.size(1) > max_src_len:
        src = src[:,:max_src_len]
        src_bert = src_bert[:,:max_src_len]
        if use_pgn:
            src_ext = src_ext[:,:max_src_len]
        if use_ner:
            src_ne = src_ne[:,:max_src_len]
    return {
        'src_raw': src_batch,
        'tgt_raw': tgt_batch,
        'src': src.to(device), 
        'tgt': tgt.to(device), 
        'src_bert': src_bert.to(device), 
        'src_ext': src_ext.to(device) if use_pgn else None,
        'tgt_ext': tgt_ext.to(device) if use_pgn else None,
        'src_ne': src_ne.to(device) if use_ner else None, 
        'dictionary_ext': dictionary_ext,
        'max_oov_len': len(dictionary_ext) - len(dictionary) if use_pgn else None
    }


def ner_for_bpe(bpe_tokens, ne_tokens, get_mask=False, special_tokens=None):
    """
    Arguments:
        bpe_tokens: list of bpe tokens (can contains bos, eos, unk token)
        ne_tokens: list of name entities of values:
            ('B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O')
        get_mask: determine return value as list of name entities or name entity mask (list of 0s and 1s)
        special_tokens: tokens in bpe_tokens but not in ne_tokens, ex: bos, eos,... 
    Return:
        if get_mask:
            extend list of name entities of same domain as ne_tokens
        else:
            list of 0s and 1s where name entities are 1s, others are 0s
    """
    idx = 0
    ne_tokens_ext = []
    if special_tokens is None:
        special_tokens = []
    for t in bpe_tokens:
        if t in special_tokens:
            value = 0 if get_mask else 'O'
        else:
            value = ne_tokens[idx]
            if get_mask:
                if value in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']:
                    value = 1
                else:
                    value = 0
            if not t.endswith("@@"):
                idx += 1
        ne_tokens_ext.append(value)
    return ne_tokens_ext
        
