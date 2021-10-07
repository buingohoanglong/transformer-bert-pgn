import math
import torch


def print_train_progress(epoch, iteration, num_iterations, loss, pbar_length=20):
    progress = math.ceil(iteration / num_iterations * pbar_length)
    pbar = "="*progress + " "*(pbar_length - progress)
    print(f'\r--|Epoch: {epoch}, progress: {iteration / num_iterations * 100:.2f}% [ {pbar} ] {iteration}/{num_iterations}, loss: {loss}',end="")


def save_checkpoint(model, optimizer, lr_scheduler, epoch, validation_bleu=None, file_dir="./", file_name='checkpoint.pt'):
    print(f'--|Saving checkpoint to {file_dir + file_name} ...')
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_scheduler_state': lr_scheduler.state_dict(),
        'validation_bleu': validation_bleu
    }
    torch.save(checkpoint, file_dir + file_name)


def load_checkpoint(file_name='checkpoint.pt', file_dir='./', device=torch.device('cpu')):
    print(f'--|Loading checkpoint from {file_dir + file_name} ...')
    return torch.load(file_dir + file_name, map_location=device)