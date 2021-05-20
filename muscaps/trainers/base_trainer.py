from abc import ABC, abstractmethod


class BaseTrainer:
    def __init__(self, model, loss, optimizer, batch_size, logger):
        pass

    @abstractmethod
    def load_dataset(self):
        """Load dataset and dataloader."""

    @abstractmethod
    def build_model(self):
        """Build the model."""

    @abstractmethod
    def load_optimizer(self):
        """Load the optimizer."""

    @abstractmethod
    def load_metrics(self):
        """Load metrics for evaluation."""

    @abstractmethod
    def train(self):
        """Run the training process."""
