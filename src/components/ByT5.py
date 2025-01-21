
from transformers import T5EncoderModel
import torch.nn as nn

class ByT5(nn.Module):
    """
    TextEncoder using ByT5 Small : 
    - This encoder converts raw text inputs into contextualized embeddings
        using the pretrained ByT5 model from HuggingFace.
    """
    def __init__(self):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained("google/byt5-small")

    def forward(self, X):
        """
        Forward pass to encode text into contextual embeddings.

        Args:
            X (list of str): A batch of text inputs.

        Returns:
            Tensor: The last hidden state from the ByT5 encoder, 
                    representing contextual embeddings for each token.
        """
        outputs = self.model(
            input_ids=X["input_ids"], 
            attention_mask=X["attention_mask"], 
        )
        
        # Return the embeddings from the last hidden state of the encoder
        return outputs.last_hidden_state
