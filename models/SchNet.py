import torch.nn as nn

from .layers import InteractionLayer, SSP


class SchNet(nn.Module):
    def __init__(self, n_node_feat, hidden_dim, n_layers, gamma, n_filters,
                 filter_spacing):
        super().__init__()
        self.node_embedding = nn.Linear(n_node_feat, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [InteractionLayer(hidden_dim, hidden_dim, hidden_dim,
                              gamma, n_filters, filter_spacing)
             for _ in range(n_layers)]
        )
        self.predict_layer1 = nn.Linear(hidden_dim, 32, bias=False)
        self.act = SSP(a=0.5, b=0.5)
        self.predict_layer2 = nn.Linear(32, 1, bias=False)

    def forward(self, sample):
        node_feat, valid, pos = self._get_needed(sample)
        _node_feat = self.node_embedding(node_feat)
        for layer in self.layers:
            _node_feat = layer(_node_feat, pos, valid)
        predict = self.predict_layer1(_node_feat)
        predict = self.act(predict)
        predict = self.predict_layer2(predict)
        predict = predict.sum(1)
        return predict

    def _get_needed(self, sample):
        values = list(sample.values())
        node_feat = values[0]
        valid = values[3]
        pos = values[4]
        return node_feat, valid, pos
