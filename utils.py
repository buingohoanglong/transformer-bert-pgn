import torch
from Dictionary import Dictionary
from torch.nn.utils.rnn import pad_sequence
import re

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
                if value in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-NOS']:
                    value = 1
                else:
                    value = 0
            if not t.endswith("@@"):
                idx += 1
        ne_tokens_ext.append(value)
    return ne_tokens_ext


def preprocess(annotator, text, ner=False):
    text = text.replace('\xa0', ' ').strip()
    sentences = annotator.ner(text) if ner else annotator.tokenize(text)
    segments = []
    for s in sentences:
        segments.extend(s)
    if len(segments) == 0:
        return {'words': [], 'name_entities': []} if ner else {'words': []}
    
    if ner:
        words = []
        name_entities = []
        signs = ['.', ',', ';', '?', '!', '(', ')', '-', '/', '\\', '\"', '%', '\'', '{', '}', '[', ']']
        for w, ne in segments:
            if w in signs or bool(re.search(r'\d', w)):
                if ne not in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']:
                    ne = 'B-NOS'
            words.append(w)
            name_entities.append(ne)
        return {
            'words': words,
            'name_entities': name_entities
        }
    else:
        return {'words': segments}
    


def build_vocab(dictionary, segmenter, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_tmp in lines:
            line = " ".join(preprocess(segmenter, line_tmp.strip())['words'])
            tokens = dictionary.tokenize(line)
            for t in tokens:
                dictionary.add_token(t)
        

def no_accent_vietnamese(s):
    s = re.sub(r'[àáạảã]', 'a', s)
    s = re.sub(r'[ấầậẩẫ]', 'â', s)
    s = re.sub(r'[ắằặẳẵ]', 'ă', s)
    s = re.sub(r'[ÀÁẠẢÃ]', 'A', s)
    s = re.sub(r'[ẦẤẬẨẪ]', 'Â', s)
    s = re.sub(r'[ẰẮẶẲẴ]', 'Ă', s)
    s = re.sub(r'[èéẹẻẽ]', 'e', s)
    s = re.sub(r'[ềếệểễ]', 'ê', s)
    s = re.sub(r'[ÈÉẸẺẼ]', 'E', s)
    s = re.sub(r'[ỀẾỆỂỄ]', 'Ê', s)
    s = re.sub(r'[òóọỏõ]', 'o', s)
    s = re.sub(r'[ồốộổỗ]', 'ô', s)
    s = re.sub(r'[ờớợởỡ]', 'ơ', s)
    s = re.sub(r'[ÒÓỌỎÕ]', 'O', s)
    s = re.sub(r'[ỒỐỘỔỖ]', 'Ô', s)
    s = re.sub(r'[ỜỚỢỞỠ]', 'Ơ', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũ]', 'u', s)
    s = re.sub(r'[ừứựửữ]', 'ư', s)
    s = re.sub(r'[ÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỪỨỰỬỮ]', 'Ư', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    return s

def format(text):
    text = re.sub(f"  ", " ", text)
    text = re.sub(f"_", " ", text)
    text = re.sub(f" \.", ".", text)
    text = re.sub(f"' ", "'", text)
    text = re.sub(f" ,", ",", text)
    text = re.sub(f" ;", ";", text)
    text = re.sub(f" %", "%", text)
    text = re.sub(f" :", ":", text)
    text = re.sub(f" !", "!", text)
    text = re.sub(f" \?", "?", text)
    text = re.sub(f"\( ", "(", text)
    text = re.sub(f" \)", ")", text)
    text = re.sub(f" / ", "/", text)
    text = re.sub(f"\" ", "\"", text)
    text = re.sub(f" \"", "\"", text)
    text = re.sub(f" g ", "g ", text)
    text = re.sub(f" kg ", "kg ", text)
    text = re.sub(f" m ", "m ", text)
    text = re.sub(f" cm ", "cm ", text)
    text = re.sub(f" m2 ", "m2 ", text)
    text = re.sub(f" ha ", "ha ", text)
    return text.strip()