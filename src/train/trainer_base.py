class Trainer:
    """
    Generic interface for a trainer. Subclasses should implement:
      - prepare_batch_data(batch)
      - train_step(batch_data)
    """

    def __init__(self, model, tokenizer, optimizer, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.verbose = verbose

    def prepare_batch_data(self, batch):
        """
        Convert raw batch data (e.g. from a data iterator) into a form needed
        for the training step. Return a 'batch_data' structure containing
        everything the trainer needs.
        """
        raise NotImplementedError

    def train_step(self, batch_data):
        """
        Execute a single training step (forward pass, compute loss, backprop).
        Return the computed loss and any other metrics of interest (e.g. reward).
        """
        raise NotImplementedError
