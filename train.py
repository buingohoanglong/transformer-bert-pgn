from Loss import Loss
from utils import *
from Dictionary import *
from model import NMT
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from Dataset import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=False)
    bert = AutoModel.from_pretrained("bert-base-multilingual-cased")

    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file('./data/vi-ba/dict.txt')
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # load train_dataset, train_loader
    train_dataset = NMTDataset('./data/vi-ba/train.vi', './data/vi-ba/train.ba')
    print(f'--|Number of train samples: {len(train_dataset)}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=4, shuffle=True)

    # load val_dataset, val_loader
    val_dataset = NMTDataset('./data/vi-ba/valid.vi', './data/vi-ba/valid.ba')
    print(f'--|Number of valid samples: {len(val_dataset)}')
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, num_workers=4, shuffle=False)

    # init criterion
    # criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token_to_index(dictionary.pad_token))
    criterion = Loss(ignore_idx=dictionary.token_to_index(dictionary.pad_token), smoothing=0.1)

    # load model
    model = NMT(
        dictionary=dictionary, 
        bert_tokenizer=bert_tokenizer, 
        annotator=annotator, 
        criterion=criterion,
        d_model=512, 
        d_ff=2048,
        num_heads=8, 
        num_layers=6, 
        dropout=0.1,
        bert=bert,
        d_bert=768,
        use_pgn=True,
        use_ner=True,
        max_src_len=512,
        max_tgt_len=512
    )
    
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # logger
    logger = TensorBoardLogger("./", name="lightning_logs", version=0)

    # init trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=64, 
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=1,
        max_epochs=21,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger
    )

    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    
if __name__ == '__main__':
    main()