import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """Here is the explanation for the code above:
    1. super().__init__() is a method that will call the __init__() method of the parent class (nn.Module).
    2. The embedding layer is a simple lookup table that stores embeddings of a fixed dictionary and size.
    3. nn.Embedding(vocab_size, d_model) will return a tensor of shape (vocab_size, d_model) where each row of the tensor is the embedding of a word in a numerical representation.
    4. self.d_model = d_model sets the dimension of the embedding vector. This will be used in the forward() method.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """__init__ method

        Args:
            vocab_size (int): vocabulary size
            d_model (int): embedding size
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """forward method

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))
