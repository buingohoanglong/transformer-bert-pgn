import torch
import torch.utils.data as tud



def beam_search(model, src, src_bert, src_ext=None, src_ne=None, max_oov_len=None, 
                beam_size=5, max_tgt_len=256, eos_idx=None):
    """
    Arguments:
        model: nn.Module
        src: [batch_size, src_seq_len]
        max_tgt_len: int
        beam_size: int
    Return:
        preds: [batch_size, beam_size, max_tgt_len]
        seq_probs: [batch_size, beam_size]
            The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
            probability of the next token at every step.
    """
    #TODO replace forward() with appropriate method
    
    with torch.no_grad():
        batch_size = src.shape[0]
        preds = torch.zeros(batch_size, 1, device=src.device).long() # [batch_size, current_len (=1)]
        next_probs = forward(src, preds)[:, -1, :] # [batch_size, vocab_size]
        vocab_size = next_probs.shape[-1]
        seq_probs, next_tokens = next_probs.squeeze().log_softmax(-1).topk(k=beam_size, axis=-1) #TODO remove softmax, add epsilon to avoid inf, [batch_size, beam_size], [batch_size, beam_size]
        preds = preds.repeat((beam_size, 1)) # [batch_size * beam_size, current_len (=1)]
        next_tokens = next_tokens.reshape(-1, 1) # [batch_size * beam_size, 1]
        preds = torch.cat((preds, next_tokens), axis=-1) # [batch_size * beam_size, current_len (=2)]
        max_tgt_len_iterator = range(max_tgt_len - 1)
        for i in max_tgt_len_iterator:
            dataset = tud.TensorDataset(src.repeat((beam_size, 1, 1)).transpose(0, 1).flatten(end_dim=1), preds) # [batch_size * beam_size, src_seq_len], [batch_size * beam_size, current_len]
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            next_probs = []
            iterator = iter(loader)
            for x, y in iterator: # [batch_size, src_seq_len], [batch_size, current_len]
                next_probs.append(forward(x, y)[:, -1, :].log_softmax(-1)) # [batch_size, vocab_size]
            next_probs = torch.cat(next_probs, axis=0) # [batch_size * beam_size, vocab_size]
            next_probs = next_probs.reshape((-1, beam_size, next_probs.shape[-1])) # [batch_size, beam_size, vocab_size]
            seq_probs = seq_probs.unsqueeze(-1) + next_probs # [batch_size, beam_size, vocab_size]
            seq_probs = seq_probs.flatten(start_dim=1) # [batch_size, beam_size * vocab_size]
            seq_probs, idx = seq_probs.topk(k=beam_size, axis=-1) # [batch_size, beam_size], [batch_size, beam_size]
            next_tokens = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1) # [batch_size * beam_size, 1]
            best_candidates = (idx / vocab_size).long() # [batch_size, beam_size]
            best_candidates += torch.arange(preds.shape[0] // beam_size).unsqueeze(-1) * beam_size # [batch_size, beam_size]
            preds = preds[best_candidates].flatten(end_dim=-2) # [batch_size * beam_size, current_len]
            preds = torch.cat((preds, next_tokens), axis=1) # [batch_size * beam_size, current_len + 1]
        return preds.reshape(-1, beam_size, preds.shape[-1]), seq_probs
