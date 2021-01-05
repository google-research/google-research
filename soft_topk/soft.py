# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""
Created on Thu Feb  6 13:02:31 2020
@author: yujia
"""

import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
# pylint: skip-file


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    """standard forward of sinkhorn."""

    bs, _, k_ = C.size()

    v = torch.ones([bs, 1, k_])/(k_)
    G = torch.exp(-C/epsilon)
    if torch.cuda.is_available():
        v = v.cuda()

    for _ in range(max_iter):
        u = mu/(G*v).sum(-1, keepdim=True)
        v = nu/(G*u).sum(-2, keepdim=True)

    Gamma = u*G*v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    """sinkhorn forward in log space."""

    bs, n, k_ = C.size()
    k = k_-1

    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, k+1])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()
    epsilon_log_mu = epsilon*torch.log(mu)
    epsilon_log_nu = epsilon*torch.log(nu)
    def min_epsilon_row(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)

    for _ in range(max_iter):
        f = min_epsilon_row(C-g, epsilon)+epsilon_log_mu
        g = min_epsilon_col(C-f, epsilon)+epsilon_log_nu

    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:,:,:-1]
    Gamma_ = Gamma[:,:,:-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1./(mu.view([1,-1]))  #[1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            -torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)   #[bs, k, k]

    inv_Kappa = torch.inverse(Kappa) #[bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1)*Gamma_
    L = Gamma_mu.matmul(inv_Kappa) #[bs, n, k]
    G1 = grad_output_Gamma * Gamma #[bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1*inv_mu).unsqueeze(-1)*Gamma  #[bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  #[bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1,-2)).transpose(-1,-2)*Gamma  #[bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode='constant', value=0)*Gamma  #[bs, n, k+1]
    G2 = G21 + G22 + G23  #[bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1) #[bs, k+1, 1]
    g2 = g2[:,:-1,:]  #[bs, k, 1]
    G31 = - L.matmul(g2)*Gamma  #[bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1,-2), pad=(0, 1), mode='constant', value=0)*Gamma  #[bs, n, k+1]
    G3 = G31 + G32  #[bs, n, k+1]
#            del g2, G31, G32, L

    grad_C = (-G1+G2+G3)/epsilon  #[bs, n, k+1]

    return grad_C


class TopKFunc1(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon>1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma!=Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        #Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter = 200):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0,1]).view([1,1, 2])
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        #find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_==float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores-min_scores)
        mask = scores==float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores-self.anchors)**2
        C = C / (C.max().detach())
        #print(C)
        mu = torch.ones([1, n, 1], requires_grad=False)/n
        nu = torch.FloatTensor([self.k/n, (n-self.k)/n]).view([1, 1, 2])

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()

        Gamma = TopKFunc1.apply(C, mu, nu, self.epsilon, self.max_iter)
        #print(Gamma)
        A = Gamma[:,:,0]*n

        return A

############################################################################
############################################################################

class TopK_stablized(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter = 200):
        super(TopK_stablized, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0,1]).view([1,2,1])
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()[:2]
        scores = scores.view([bs, 1, n])

        #find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_==float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores-min_scores)
        mask = scores==float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores-self.anchors)**2
        C = C / (C.max().detach())
        f = torch.zeros([bs, 1, n])
        g = torch.zeros([bs, 2, 1])
        mu = torch.ones([1, 1, n], requires_grad=False)/n
        nu = torch.FloatTensor([self.k/n, (n-self.k)/n]).view([1, 2, 1])

        if torch.cuda.is_available():
            f = f.cuda()
            g = g.cuda()
            mu = mu.cuda()
            nu = nu.cuda()

        def min_epsilon_row(Z, epsilon):
            return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)


        def min_epsilon_col(Z, epsilon):
            return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)


        for i in range(self.max_iter):
            f = min_epsilon_col(C-f-g, self.epsilon)+f+self.epsilon*torch.log(mu)
            g = min_epsilon_row(C-f-g, self.epsilon)+ g +self.epsilon*torch.log(nu)

        P = torch.exp((-C+f+g)/self.epsilon)
        A = P[:,0,:]*n
        return A

