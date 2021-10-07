import torch
import math
from utils import *
from Dictionary import *
from torch.nn.utils.rnn import pad_sequence


def train(model, criterion, optimizer, lr_scheduler, dataloader, dictionary, tokenizer, segmenter,
        num_epochs=None, init_epoch=1, accumulation_factor=1, use_pgn=False):
    # total_samples = len(dataloader.dataset)
    # batch_size = dataloader.batch_size
    num_batch = len(dataloader)
    # num_update = math.ceil(total_samples / (batch_size * accumulation_factor))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    print(f'--|Device: {device}')

    epoch = init_epoch
    while True:
        for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
            # prepare input
            src, tgt, src_bert, src_ext, dictionary_ext = process_batch(
                src_batch, tgt_batch, dictionary, tokenizer, segmenter, use_pgn=use_pgn
            )
            src = src.to(device)
            tgt = tgt.to(device)
            src_bert = src_bert.to(device)
            max_oov_len = None
            if use_pgn:
                src_ext = src_ext.to(device)
                max_oov_len = dictionary_ext.vocab_size - dictionary.vocab_size

            # forward pass
            output = model(src, tgt, src_bert, src_ext, max_oov_len)  # [batch_size, seq_len, vocab_size] 

            # compute loss
            loss = criterion(output.transpose(1,2), tgt) 
            loss = loss / accumulation_factor

            # backward pass
            loss.backward()
            if (batch_idx + 1) % accumulation_factor == 0 or (batch_idx + 1) == num_batch:
                # update weight
                optimizer.step()
                optimizer.zero_grad()

            # print train progress
            print_train_progress(epoch, batch_idx+1, num_batch, loss.detach().item() * accumulation_factor)

        # update learning rate
        lr_scheduler.step()

        # Save checkpoint
        save_checkpoint(model, optimizer, lr_scheduler, epoch, file_dir='./checkpoints/', file_name=f'checkpoint{epoch}.pt')

        epoch += 1
        if num_epochs is not None and epoch > num_epochs:
            break



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