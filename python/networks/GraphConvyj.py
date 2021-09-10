import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform

def mesh_mean_pooling(input, mapping, degree):
    padding_feature = torch.zeros(input.shape[0], 1, input.shape[2], dtype=torch.float32).to(torch.cuda.current_device())
    padded_input = torch.cat([padding_feature, input], 1)

    total_nb_feature = torch.zeros(input.shape[0], mapping.shape[0], mapping.shape[1], input.shape[2], dtype=torch.float32).to(torch.cuda.current_device())
    for i in range(mapping.shape[1]):
        total_nb_feature[:, :, i, :] = padded_input[:, mapping[:, i], :]
    mean_nb_feature = total_nb_feature.sum(dim=2)/degree
    return mean_nb_feature

def mesh_mean_depooling(input, mapping, degree):
    padded_input = input
    # padding_feature = torch.zeros(input.shape[0], 1, input.shape[2], dtype=torch.float32).to(torch.cuda.current_device())
    # padded_input = torch.cat([padding_feature, input], 1)
    total_nb_feature = torch.zeros(input.shape[0], mapping.shape[0], mapping.shape[1], input.shape[2], dtype=torch.float32).to(torch.cuda.current_device())
    for i in range(mapping.shape[1]):
        total_nb_feature[:, :, i, :] = padded_input[:, mapping[:, i], :]
    mean_nb_feature = total_nb_feature.sum(dim=2)/degree
    return mean_nb_feature


class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}_2 \mathbf{x}_j.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        h = torch.matmul(x, self.weight).transpose(0, 1)
        x = x.transpose(0, 1)
        out = self.propagate(edge_index, size=size, x=x, h=h, edge_weight=edge_weight)
        out = out.transpose(0, 1)  # out has shape [batch_size, num_nodes, num_features]
        return out


    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

from torch_geometric.nn.inits import glorot, zeros

class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(0)} &= \mathbf{X}

        \mathbf{Z}^{(1)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a scalar when
            operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, K, edge_index, normalization='sym',
                 bias=True, lambda_max=None, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        self.edge_index, self.norm = self.norm_(edge_index, torch.max(edge_index)+1, None,
                                     self.normalization, lambda_max,
                                     dtype=torch.float32, batch=None)

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    # @staticmethod
    def norm_(self, edge_index, num_nodes, edge_weight, normalization, lambda_max,
             dtype=None, batch=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes)

        if batch is not None and torch.is_tensor(lambda_max):
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight[edge_weight == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1, num_nodes=num_nodes)

        return edge_index, edge_weight


    # def forward(self, x, edge_index, edge_weight=None, batch=None,
    #             lambda_max=None):
    #     """"""
    #     if self.normalization != 'sym' and lambda_max is None:
    #         raise ValueError('You need to pass `lambda_max` to `forward() in`'
    #                          'case the normalization is non-symmetric.')
    #     lambda_max = 2.0 if lambda_max is None else lambda_max

    #     edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
    #                                  self.normalization, lambda_max,
    #                                  dtype=x.dtype, batch=batch)

    #     Tx_0 = x
    #     out = torch.matmul(Tx_0, self.weight[0])

    #     if self.weight.size(0) > 1:
    #         Tx_1 = self.propagate(edge_index, x=x, norm=norm)
    #         out = out + torch.matmul(Tx_1, self.weight[1])

    #     for k in range(2, self.weight.size(0)):
    #         Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
    #         out = out + torch.matmul(Tx_2, self.weight[k])
    #         Tx_0, Tx_1 = Tx_1, Tx_2

    #     if self.bias is not None:
    #         out = out + self.bias

    #     return out

    def forward(self, x, edge_weight=None):
        """"""

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        x = x.transpose(0,1)
        x = x.reshape(-1, x.shape[-2]*x.shape[-1])
        Tx_0 = x
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(self.edge_index, x=x, norm=self.norm)
            Tx_1_reshaped = Tx_1.reshape(Tx_1.shape[0],-1,self.in_channels).transpose(0,1)
            out = out + torch.matmul(Tx_1_reshaped, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(self.edge_index, x=Tx_1, norm=self.norm) - Tx_0
            Tx_2_reshaped = Tx_2.reshape(Tx_2.shape[0],-1,self.in_channels).transpose(0,1)
            out = out + torch.matmul(Tx_2_reshaped, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return self.norm.view(-1, 1) * x_j
    # def message(self, x_j, norm):
    #     res = self.norm.view(-1, 1) * x_j.transpose(0,1)
    #     return res.transpose(0,1)

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)



from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, edge_index, improved=True, cached=False,
                 bias=True, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.cached_num_edges = edge_index.size(1)
        edge_index, norm = self.norm_(edge_index, torch.max(edge_index)+1, None,
                                     self.improved, torch.float32)
        self.cached_result = edge_index, norm
        self.edge_index, self.norm = self.cached_result

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    # @staticmethod
    def norm_(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        # edge_weight torch.Size([57600])
        fill_value = 1 if not improved else 2

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        # edge_weight torch.Size([67202])
        # edge_index torch.Size([2, 67202])
        row, col = edge_index
        # row, col: torch.Size([67202])
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # deg: torch.Size([9602])
        deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt: torch.Size([9602])
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # deg_inv_sqrt: torch.Size([9602])

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_weight=None):
        """"""
        # x torch.Size([16, 9602, 9])
        # self.weight torch.Size([9, 9])
        x = torch.matmul(x, self.weight)
        # self.edge_index = self.edge_index.cuda(self.weight.device)
        self.edge_index = self.edge_index.to(self.weight.device)
        # self.norm torch.Size([67202])
        # self.norm = self.norm.cuda(self.weight.device)
        self.norm = self.norm.to(self.weight.device)

        x = x.transpose(0, 1)  # x has shape [num_nodes, batch_size, num_features]
        out = self.propagate(self.edge_index, x=x, norm=self.norm)
        out = out.transpose(0, 1)  # out has shape [batch_size, num_nodes, num_features]

        return out


    def message(self, x_j, norm):
        res = self.norm.view(-1, 1) * x_j.transpose(0,1)
        return res.transpose(0,1)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



from torch_geometric.nn.inits import glorot, zeros


class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.

    :rtype: :class:`Tensor`
    """
    def __init__(self, in_channels, out_channels, adj, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        # adj prealloc
        add_loop = True
        N, _ = adj.size()
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        self.adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj = None, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        B, N, _ = x.size()
        # print(self.adj.size())
        adj = self.adj.repeat(B, 1, 1)

        x = x.unsqueeze(0) if x.dim() == 2 else x

        out = torch.matmul(x, self.weight)

        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)