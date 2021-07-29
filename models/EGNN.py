import torch.nn as nn

from .layers import EGCL


class EGNN(nn.Module):
    def __init__(self, n_node_feat, hidden_dim, n_layers, infer=False):
        super().__init__()
        self.node_embedding = nn.Linear(n_node_feat, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [EGCL(hidden_dim, hidden_dim, infer, True) for _ in range(n_layers)]
        )
        self.predict_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.predict_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, node_feat, pos, valid, adj):
        node_feat = self.node_embedding(node_feat)
        for layer in self.layers:
            pos, node_feat = layer(node_feat, pos, valid, adj)
        predict = self.predict_layer1(node_feat)
        predict = predict.sum(1)
        predict = self.predict_layer2(predict)
        return predict
