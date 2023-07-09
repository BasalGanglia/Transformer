import torch
import unittest
from src.transformer.model import (
    InputEmbeddings,
)  # replace 'your_module' with the actual module name


class TestInputEmbeddings(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10
        self.d_model = 5
        self.input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)

    def test_forward_output_shape(self):
        x = torch.tensor([1, 2, 3])
        output = self.input_embeddings.forward(x)
        self.assertEqual(output.shape, (3, self.d_model))

    def test_forward_output_values(self):
        x = torch.tensor([0])  # testing with an input tensor of size 1 for simplicity
        output = self.input_embeddings.forward(x)
        expected_output = self.input_embeddings.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model)
        )
        self.assertTrue(torch.allclose(output, expected_output))
