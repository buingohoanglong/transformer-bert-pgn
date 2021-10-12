import torch.nn as nn
from utils import *
from Dictionary import *
from model import Transformer, NMT
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from Dataset import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# def train(model, criterion, optimizer, lr_scheduler, dataloader, dictionary, tokenizer, segmenter,
#         num_epochs=None, init_epoch=1, accumulation_factor=1, use_pgn=False):
#     # total_samples = len(dataloader.dataset)
#     # batch_size = dataloader.batch_size
#     num_batch = len(dataloader)
#     # num_update = math.ceil(total_samples / (batch_size * accumulation_factor))

#     device = torch.device('cpu')
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     model.to(device)
#     print(f'--|Device: {device}')

#     epoch = init_epoch
#     while True:
#         for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
#             # prepare input
#             src, tgt, src_bert, src_ext, dictionary_ext = process_batch(
#                 src_batch, tgt_batch, dictionary, tokenizer, segmenter, use_pgn=use_pgn
#             )
#             src = src.to(device)
#             tgt = tgt.to(device)
#             src_bert = src_bert.to(device)
#             max_oov_len = None
#             if use_pgn:
#                 src_ext = src_ext.to(device)
#                 max_oov_len = dictionary_ext.vocab_size - dictionary.vocab_size

#             # forward pass
#             output = model(src, tgt, src_bert, src_ext, max_oov_len)  # [batch_size, seq_len, vocab_size] 

#             # compute loss
#             loss = criterion(output.transpose(1,2), tgt) 
#             loss = loss / accumulation_factor

#             # backward pass
#             loss.backward()
#             if (batch_idx + 1) % accumulation_factor == 0 or (batch_idx + 1) == num_batch:
#                 # update weight
#                 optimizer.step()
#                 # update learning rate
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             # print train progress
#             print_train_progress(epoch, batch_idx+1, num_batch, loss.detach().item() * accumulation_factor)

#         # Save checkpoint
#         save_checkpoint(model, optimizer, lr_scheduler, epoch, file_dir='./checkpoints/', file_name=f'checkpoint{epoch}.pt')

#         epoch += 1
#         if num_epochs is not None and epoch > num_epochs:
#             break


def main():
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")

    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file('./data/vi-ba/dict.txt')
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # load train_dataset, train_loader
    train_dataset = NMTDataset('./data/vi-ba/train_dev.vi', './data/vi-ba/train_dev.ba')
    print(f'--|Number of train samples: {len(train_dataset)}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=4)

    # load val_dataset, val_loader
    val_dataset = NMTDataset('./data/vi-ba/valid_dev.vi', './data/vi-ba/valid_dev.ba')
    print(f'--|Number of valid samples: {len(val_dataset)}')
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)

    # init criterion
    criterion = nn.CrossEntropyLoss()

    # load model
    model = Transformer(
        vocab_size=len(dictionary),
        d_model=512, 
        d_ff=2048,
        num_heads=8, 
        num_layers=6, 
        dropout=0.1,
        bert=phobert,
        d_bert=768, 
        padding_idx=dictionary.token_to_index(dictionary.pad_token),
        use_pgn=True,
        unk_idx=dictionary.token_to_index(dictionary.unk_token)
    )
    # print(model)
    
    task = NMT(
        model=model, 
        dictionary=dictionary, 
        tokenizer=tokenizer, 
        segmenter=rdrsegmenter, 
        criterion=criterion
    )

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # init trainer
    # trainer = pl.Trainer(
    #     accumulate_grad_batches=5, 
    #     gpus=1 if torch.cuda.is_available() else 0,
    #     log_every_n_steps=1,
    #     callbacks=[checkpoint_callback]
    # )
    trainer = pl.Trainer(
        resume_from_checkpoint="./checkpoints/checkpoint-epoch=06-val_loss=8.61.ckpt",
        callbacks=[checkpoint_callback]
    )

    # train
    trainer.fit(model=task, train_dataloaders=train_loader, val_dataloaders=val_loader)

    
if __name__ == '__main__':
    main()