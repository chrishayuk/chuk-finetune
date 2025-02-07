# src/train/trainer.py
class Trainer:
    """
    Generic interface for a trainer. Subclasses must implement:
      - prepare_batch_data(batch)
      - train_step(batch_data)
    Optionally override:
      - on_batch_start(batch_idx)
      - on_batch_end(batch_idx, loss, reward)
      - on_epoch_start(epoch)
      - on_epoch_end(epoch, mean_loss, mean_reward)
    """

    def __init__(self, model, tokenizer, optimizer, device=None, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose

    def prepare_batch_data(self, batch):
        raise NotImplementedError()

    def train_step(self, batch_data):
        raise NotImplementedError()

    def on_batch_start(self, epoch, batch_idx):
        pass

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, mean_loss, mean_reward):
        pass
