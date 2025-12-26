import torch
from torch import nn


class BaselineClassifier(nn.Module):
    "Fully-connected baseline text classifier"

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        emb_dim: int,
        pad_idx: int,
        hidden_dim: int,
        n_classes: int,
    ) -> "BaselineClassifier":
        """Init model.

        Args:
            seq_len (int): input sequence length in tokens.
            vocab_size (int): size of the tokenizer vocabulary.
            emb_dim (int): embedding vector length.
            pad_idx (int): Padding token index. Embeddings for
                this index are initialized to zero and remain
                zero during training.
            hidden_dim (int): dimension of linear layers.
            n_classes (int): number of classes in classification task.
        """
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=seq_len, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): input tensor with shape (batch_size, seq_len, emb_dims).
        """
        x = self.embedding(x)
        return self.fc_layers(x)
