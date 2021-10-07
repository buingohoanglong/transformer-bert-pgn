from torch.optim.lr_scheduler import _LRScheduler

class NoamLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, d_model):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        lr = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [lr for base_lr in self.base_lrs]