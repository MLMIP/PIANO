import torch
from torch import nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim, out_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, out_dim)
        self.f = nn.Softmax(dim=0)
        self.out_dim = out_dim

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        # softmax = nn.functional.softmax
        # att_w = nn.Softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        att_w = self.f(self.W(batch_rep))
        utter_rep = torch.sum(batch_rep * att_w, dim=0)

        return utter_rep.view(1, self.out_dim)
