import torch
import torch.nn.functional as F
import torch.nn as nn

class SimpleFFN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_size_per_expert=0,
        activation_fn=None,
        activation_fn_with_self=None,
        idln=False
    ):
        super().__init__()
        self.skip_expert = int(torch.os.environ.get("SKIP_EXPERT", "0")) != 0
        self.hidden_size_per_expert = hidden_size_per_expert
        self.output_dim = output_dim or input_dim
        self.idln = idln
        if self.idln:
            self.ln = nn.LayerNorm(self.output_dim)
        #print('Hidden size per expert accept :', self.hidden_size_per_expert)
        if activation_fn_with_self is not None:
            assert (
                activation_fn is None
            ), "Option `activation_fn_with_self` has been specified, please keep exactly one of them."
            activation_fn = lambda x: activation_fn_with_self(x, self)
        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        self.activation_fn = activation_fn
        if hidden_size_per_expert:
            self.fc1 = nn.Linear(input_dim, hidden_size_per_expert)
            self.fc2 = nn.Linear(hidden_size_per_expert, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
    
        if self.hidden_size_per_expert:
            out = self.activation_fn(out)
            out = self.fc2(out)

        return self.ln(out) if self.idln else out
