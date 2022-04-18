from embedding import *
from hyper_embedding import *
from collections import OrderedDict
import torch
import json
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num, norm):
        norm = norm[:,:1,:,:]						# revise
        h = h - torch.sum(h * norm, -1, True) * norm
        t = t - torch.sum(t * norm, -1, True) * norm
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class MetaR(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, embed = None):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.embedding = Embedding(dataset, parameter)
        self.h_embedding = H_Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.h_emb = nn.Embedding(self.num_rel, self.embed_dim)
        init.xavier_uniform_(self.h_emb.weight)

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.attn_w = nn.Linear(self.embed_dim, 1)

        self.gate_w = nn.Linear(self.embed_dim, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_w.weight)

        self.symbol_emb.weight.requires_grad = False
        self.h_norm = None

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = LSTM_attn(embed_size=50, n_hidden=100, out_size=50,layers=2)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=450, out_size=100, layers=2)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.norm_q_sharing = dict()


    def neighbor_encoder(self, connections, num_neighbors, istest):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        entity_self = connections[:,:,0].squeeze(-1)
        relations = connections[:,:,1].squeeze(-1)
        entities = connections[:,:,2].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)
        entself_embeds = self.dropout(self.symbol_emb(entself))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = F.leaky_relu(out)
        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        out_attn = out_attn.squeeze(2)
        gate_tmp = self.gate_w(out_attn) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        out_neigh = torch.mul(out_attn, gate)
        out_neighbor = out_neigh + torch.mul(entself_embeds,1.0-gate)

        return out_neighbor

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2


    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        norm_vector = self.h_embedding(task[0])
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[0]
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, istest)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, istest)
        support_few = torch.cat((support_left, support_right), dim=-1)
        support_few = support_few.view(support_few.shape[0], 2, self.embed_dim)

        for i in range(self.few-1):
            support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[i+1]
            support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, istest)
            support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, istest)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], 2, self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        support_few = support_few.view(support_few.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]

        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, norm_vector)	# revise norm_vector

                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
                norm_q = norm_vector - self.beta*grad_meta				# hyper-plane update
            else:
                rel_q = rel
                norm_q = norm_vector

            self.rel_q_sharing[curr_rel] = rel_q
            self.h_norm = norm_vector.mean(0)
            self.h_norm = self.h_norm.unsqueeze(0)

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        if iseval:
            norm_q = self.h_norm
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_q)

        return p_score, n_score
