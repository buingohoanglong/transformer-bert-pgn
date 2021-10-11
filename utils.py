import math
import torch
from Dictionary import Dictionary, preprocess
from torch.nn.utils.rnn import pad_sequence

# def print_train_progress(epoch, iteration, num_iterations, loss, pbar_length=20):
#     progress = math.ceil(iteration / num_iterations * pbar_length)
#     pbar = "="*progress + " "*(pbar_length - progress)
#     print(f'\r--|Epoch: {epoch}, progress: {iteration / num_iterations * 100:.2f}% [ {pbar} ] {iteration}/{num_iterations}, loss: {loss}',end="")


# def save_checkpoint(model, optimizer, lr_scheduler, epoch, validation_bleu=None, file_dir="./", file_name='checkpoint.pt'):
#     print(f'--|Saving checkpoint to {file_dir + file_name} ...')
#     checkpoint = {
#         'epoch': epoch,
#         'model_state': model.state_dict(),
#         'optimizer_state': optimizer.state_dict(),
#         'lr_scheduler_state': lr_scheduler.state_dict(),
#         'validation_bleu': validation_bleu
#     }
#     torch.save(checkpoint, file_dir + file_name)


# def load_checkpoint(file_name='checkpoint.pt', file_dir='./', device=torch.device('cpu')):
#     print(f'--|Loading checkpoint from {file_dir + file_name} ...')
#     return torch.load(file_dir + file_name, map_location=device)


def process_batch(src_batch, tgt_batch, dictionary, tokenizer, segmenter, max_src_seq_length=256, use_pgn=False):
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
    for idx in range(len(src_batch)):
        src_str = preprocess(segmenter, src_batch[idx].strip())
        tgt_str = preprocess(segmenter, tgt_batch[idx].strip())
        src.append(dictionary.encode(src_str, append_bos=False, append_eos=True))
        tgt.append(dictionary.encode(tgt_str, append_bos=True, append_eos=True))
        src_bert.append(torch.tensor(tokenizer.encode(src_str)[1:]))
        if use_pgn:
            src_ext.append(dictionary_ext.encode(src_str, append_bos=False, append_eos=True, update=True))

    src = pad_sequence(src, padding_value=dictionary.token_to_index(dictionary.pad_token)).transpose(0,1)
    tgt = pad_sequence(tgt, padding_value=dictionary.token_to_index(dictionary.pad_token)).transpose(0,1)
    src_bert = pad_sequence(src_bert, padding_value=tokenizer.pad_token_id).transpose(0,1)
    if use_pgn:
        src_ext = pad_sequence(src_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token)).transpose(0,1)
    assert src.size(1) == src_bert.size(1)
    # Truncate if seq_len exceed max_src_seq_length
    if src.size(1) > max_src_seq_length:
        src = src[:,:max_src_seq_length]
        src_bert = src_bert[:,:max_src_seq_length]
        if use_pgn:
            src_ext = src_ext[:,:max_src_seq_length]
    return src, tgt, src_bert, src_ext, dictionary_ext