import torch.nn as nn

from layers import EGCL


class EGNN(nn.Module):
    def __init__(
        self,
        n_node_feat,
        hidden_dim,
        n_layer,
        n_edge_feat=1,
        infer=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EGCL(
                    n_node_feat,
                    hidden_dim,
                    n_edge_feat,
                    infer,
                )
                for _ in range(n_layer)
            ]
        )

        self.predict_layer1 = nn.Sequential(
            nn.Linear(n_node_feat, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predict_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, pos, valid, adj):
        for layer in self.layers:
            pos, node_feat = layer(node_feat, pos, valid, adj)
        predict = self.predict_layer1(node_feat)
        predict = predict.sum(1)
        predict = self.predict_layer2(predict)
        return predict
