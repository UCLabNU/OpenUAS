import torch.nn as nn
import torch


class Area2Vec(nn.Module):
    def __init__(self, num_areas, embed_size, num_output_tokens, device = "cpu"):
        """
        Initializes the Area2Vec model.

        Parameters:
        - num_areas: Total number of distinct areas.
        - embed_size: Size of the embedding vector for each area.
        - num_output_tokens: Number of tokens in the decoded output.
        """
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(num_areas, embed_size)
        self.decode_linear = nn.Linear(
            embed_size, num_output_tokens, bias=False)

    def initialize_weights(self, embedding_weight=None, decoder_weight=None, freeze_anchor_num=0):
        """
        Initializes weights for the layers in the model.
        
        Parameters:
        - embedding_weight: A tensor containing the initial weights for the embedding layer.
        - decoder_weight: A tensor containing the initial weights for the decoder layer.
        - freeze_anchor_num: The number of embeddings to freeze (optional).
        """
        if embedding_weight is not None:
            # Assign pre-initialized weights
            self.embedding.weight.data = embedding_weight.clone().detach()
            # Freeze the specified number of embeddings
            if freeze_anchor_num > 0:
                self.embedding.weight.requires_grad = True  # Enable gradients for all weights
                # Disable gradients for the last 'freeze_anchor_num' embeddings
                def _backward_hook(grad):
                    grad[-freeze_anchor_num:] = 0  # Set gradients of last 'freeze_anchor_num' weights to 0
                    return grad

                # Register the backward hook
                self.embedding.weight.register_hook(_backward_hook)

        else:
            # Initialize to a uniform distribution as before
            self.embedding.weight.data.uniform_(-0.5/self.embedding.embedding_dim,
                                                0.5/self.embedding.embedding_dim)
        
        if decoder_weight is not None:
            self.decode_linear.weight.data = decoder_weight.clone().detach()
        else:
            self.decode_linear.weight.data.uniform_(0.0, 0.0)
        
    def load_model(self, path):
        """
        Load model weights from a given path.

        Parameters:
        - path: Path to the saved model weights.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, x):
        """
        Parameters:
        - x: Input tokens representing areas.
        Returns:
        - Output after embedding and linear decoding.
        """
        hidden = self.embedding(x)
        return self.decode_linear(hidden)
