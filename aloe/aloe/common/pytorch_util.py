# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from aloe.common.consts import t_float


def pad_sequence(sequences, max_len=None, batch_first=False, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, Ellipsis] = tensor
        else:
            out_tensor[:length, i, Ellipsis] = tensor

    return out_tensor


def _glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        _glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        _glorot_uniform(m.weight.data)
    elif isinstance(m, nn.Embedding):
        _glorot_uniform(m.weight.data)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def glorot_uniform(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList) or isinstance(p, nn.ModuleList):
            for pp in p:
                _param_init(pp)
        elif isinstance(p, nn.ParameterDict) or isinstance(p, nn.ModuleDict):
            for key in p:
                _param_init(p[key])
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))

        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]

        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x


class TreeLSTMCell(nn.Module):
    def __init__(self, arity, latent_dim):
        super(TreeLSTMCell, self).__init__()
        self.arity = arity
        self.latent_dim = latent_dim

        self.mlp_i = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='sigmoid')

        self.mlp_o = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='sigmoid')

        self.mlp_u = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='tanh')

        f_list = []
        for _ in range(arity):
            mlp_f = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='tanh')
            f_list.append(mlp_f)
        self.f_list = nn.ModuleList(f_list)

    def forward(self, list_h_mat, list_c_mat):
        assert len(list_c_mat) == self.arity == len(list_h_mat)
        h_mat = torch.cat(list_h_mat, dim=-1)
        assert h_mat.shape[1] == self.arity * self.latent_dim

        i_j = self.mlp_i(h_mat)

        f_sum = 0
        for i in range(self.arity):
            f = self.f_list[i](h_mat)
            f_sum = f_sum + f * list_c_mat[i]

        o_j = self.mlp_o(h_mat)

        u_j = self.mlp_u(h_mat)

        c_j = i_j * u_j + f_sum

        h_j = o_j * torch.tanh(c_j)

        return h_j, c_j


class BinaryTreeLSTMCell(TreeLSTMCell):
    def __init__(self, latent_dim):
        super(BinaryTreeLSTMCell, self).__init__(2, latent_dim)

    def forward(self, lch_state, rch_state):
        list_h_mat, list_c_mat = zip(lch_state, rch_state)
        return super(BinaryTreeLSTMCell, self).forward(list_h_mat, list_c_mat)


class MaskedEmbedFunc(Function):
    @staticmethod
    def forward(ctx, indices, embedding, masked_token):
        out_shape = indices.shape + (embedding.shape[-1],)
        out = embedding.new(out_shape).zero_().view(-1, embedding.shape[-1])

        flat_idx = indices.view(-1)
        mask = flat_idx != masked_token
        out[mask] = embedding[flat_idx[mask]]
        ctx.flat_idx = flat_idx
        ctx.mask = mask
        ctx.embed_shape = embedding.shape
        return out.view(out_shape)

    @staticmethod
    def backward(ctx, grad_output):
        embed_shape = ctx.embed_shape
        flat_idx = ctx.flat_idx
        mask = ctx.mask
        out = grad_output.new(embed_shape).zero_()
        grad_output = grad_output.contiguous().view(-1, embed_shape[1])
        out[flat_idx[mask]] = grad_output[mask]
        return None, out, None


class MaskedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, masked_token):
        super(MaskedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.masked_token = masked_token
        self.embed = nn.Parameter(torch.Tensor(vocab_size, embed_dim))
        glorot_uniform(self)

    def forward(self, indices):
        return MaskedEmbedFunc.apply(indices, self.embed, self.masked_token)


class PosEncoding(nn.Module):
    def __init__(self, dim, base=10000, bias=0):
        super(PosEncoding, self).__init__()

        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.dim = dim
        self.register_buffer('sft', torch.tensor(sft, dtype=t_float).view(1, -1))
        self.register_buffer('base', torch.tensor(p, dtype=t_float).view(1, -1))

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=t_float).to(self.base.device)
            out_shape = pos.shape + (self.dim,)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x).view(out_shape)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def hamming_mmd(x, y):
    x = x.float()
    y = y.float()
    with torch.no_grad():
        kxx = torch.mm(x, x.transpose(0, 1))
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = torch.mm(y, y.transpose(0, 1))
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(torch.mm(y, x.transpose(0, 1))) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd


def get_gamma(X, bandwidth):
    with torch.no_grad():
        x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        x_t = torch.transpose(X, 0, 1)
        x_norm_t = x_norm.view(1, -1)
        t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
        dist2 = F.relu(Variable(t)).detach().data

        d = dist2.cpu().numpy()
        d = d[np.isfinite(d)]
        d = d[d > 0]
        median_dist2 = float(np.median(d))
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma


def get_kernel_mat(x, landmarks, gamma):
    d = pairwise_distances(x, landmarks)
    k = torch.exp(d * -gamma)
    k = k.view(x.shape[0], -1)
    return k


def MMD(x, y, bandwidth=1.0):
    y = y.detach()
    gamma = get_gamma(x.detach(), bandwidth)
    kxx = get_kernel_mat(x, x, gamma)
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    kyy[idx, idx] = 0.0
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd


if __name__ == '__main__':
    # test_multi_select()

    x = torch.randint(0, 2, size=(10, 3))
    y = torch.randint(0, 2, size=(5, 3))

    print(hamming_mmd(x, y))

    x = torch.randn(10, 2)
    y = torch.randn(5, 2)

    print(MMD(x, y))
