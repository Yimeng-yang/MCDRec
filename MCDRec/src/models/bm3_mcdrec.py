# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from models.Unet import UNet
from models.diffusion import diffusion
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood

class bm3_mcdrec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(bm3_mcdrec, self).__init__(config, dataset)
        self.config = config
        self.embedding_dim = config['embedding_size']  # 64
        self.feat_embed_dim = config['embedding_size']  # 64
        self.n_layers = config['n_layers']  # 1
        self.reg_weight = config['reg_weight']  # 0.1
        self.cl_weight = config['cl_weight']  # 2
        self.dropout = config['dropout']  # 0.3
        self.dropout_prob = 0.1  # 0.1 0.2 0.5
        self.diff_weight = config['diff_weight'] 
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.dropout_adj = config['dropout_adj'] 
        self.item_adj = None
        self.mf_weight = config['mf_weight']
        self.temperature = config['temperature']
        self.w = config['w']
        self.n_nodes = self.n_users + self.n_items  # 26495 = 19445+7050

        # load dataset info
        #         ipdb.set_trace()
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).cuda()

        # 去噪
        self.masked_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.cuda()
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).cuda()
        
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)  # 19445x64
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)  # 7050x64
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        # 导入unet与diffusion模型

        self.model = UNet(self.config)
        self.diff = diffusion(self.config)
        self.steps = config['timesteps']
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        nn.init.xavier_normal_(self.predictor.weight)
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)  # 7050x4096
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian(image_adj)
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()
            #             self.image_embedding = nn.Embedding.from_pretrained(self.pretrained_item_embedding_weights, freeze=False)

            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)  # 4096——>64
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()
#             self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)  # 7050x384
            #             self.text_embedding = nn.Embedding.from_pretrained(self.pretrained_item_embedding_weights, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)  # 384——>64
            nn.init.xavier_normal_(self.text_trs.weight)

        # 设计MLP 包括不同层数，不同dropout 直接将id embedding映射成内容特征，看看内容特征有没有被充分利用
        self.mlp_1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64, self.feat_embed_dim)
        )

#         ipdb.set_trace()
        nn.init.xavier_normal_(self.mlp_1[0].weight)  # 初始化第一个线性层
        nn.init.xavier_normal_(self.mlp_1[3].weight)  # 初始化第二个线性层
        nn.init.xavier_normal_(self.mlp_1[6].weight)  # 初始化最后一个线性层

        
        self.sample_x = torch.randn_like(self.item_id_embedding.weight)
        self.sample_t = torch.randn_like(self.item_id_embedding.weight)
        self.sample_v = torch.randn_like(self.item_id_embedding.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        #         ipdb.set_trace()
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    
    
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values
    
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values
    
    
    
    def pre_epoch_processing(self, epoch_idx):
#         ipdb.set_trace()
        if self.dropout_adj <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout_adj))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m_dm(keep_indices, torch.Size((self.n_users, self.n_items)), epoch_idx)
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)
      
    
    def _normalize_adj_m_dm(self, indices, adj_size, epoch_idx):
#         ipdb.set_trace()
#         adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        with torch.no_grad():
            if epoch_idx ==0:
                item_embs = self.item_id_embedding.weight
            else:
                item_embs = self.sample_x
            user_embs = torch.nn.functional.normalize(self.user_embedding.weight.detach(), p=2, dim=-1)
            item_embs = torch.nn.functional.normalize(item_embs.detach(), p=2, dim=-1)

            scores = torch.matmul(user_embs, item_embs.transpose(0, 1))
            edge_scores = scores[indices[0], indices[1]]
            edge_scores = edge_scores * self.temperature

        
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]) + edge_scores.detach(), adj_size)
        
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values        
    

    def forward(self, adj, predicted_x):
#         ipdb.set_trace()
        h = self.item_id_embedding.weight

        t_feat_online = self.text_trs(self.text_embedding.weight)
        v_feat_online = self.image_trs(self.image_embedding.weight)

        
#         for i in range(self.n_layers):
#             h = torch.mm(self.item_adj, h)
        
    
        predicted_x = self.w * predicted_x + (1 - self.w)* h
        i_inputs = torch.cat((predicted_x, t_feat_online, v_feat_online), dim=1)
        i_inputs = self.mlp_1(i_inputs)

        #         ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)  
        ego_embeddings = torch.cat((self.user_embedding.weight, i_inputs), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)  # 26495x64
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)


        i_g_embeddings = i_g_embeddings + h

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interactions):
        # online network
#         ipdb.set_trace()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)

        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        t= torch.randint(low=0, high=self.steps, size=(t_feat_online.shape[0] // 2 + 1,)).cuda()
        t = torch.cat([t, self.steps - t - 1], dim=0)[:t_feat_online.shape[0]]

        diff_loss, predicted_x = self.diff.p_losses(self.model, self.item_id_embedding.weight, t_feat_online, v_feat_online, t, noise=None, loss_type="l2")



        u_online_ori, i_online_ori = self.forward(self.masked_adj, predicted_x)

        #         i_inputs = torch.cat((i_online_ori, t_feat_online, v_feat_online), dim=1)
        #         i_inputs = self.mlp_1(i_inputs)

        h = self.item_id_embedding.weight
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            #             i_inputs_target = i_inputs.clone()
            h_target = h.clone()
            h = h.clone()
            h.detach()
            h_target = F.dropout(h_target, self.dropout)

            u_target.detach()
            i_target.detach()

            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            #             i_inputs_target = F.dropout(i_inputs_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        h = self.predictor(h)

        users, items = interactions[0], interactions[1]
        neg_items = interactions[2]
        i_neg_online = i_online[neg_items, :]
        u_online = u_online[users, :]  # 2048x64
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]
        h_target = h_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]

            # L(align)
            loss_t = 1 - cosine_similarity(t_feat_online, h_target.detach(), dim=-1).mean()
            # L(mask)
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, h_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

            # L(rec)
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()
        
        batch_mf_loss = self.bpr_loss(u_online, i_online, i_neg_online)

        # 再加上正则化损失(惩罚)
#         ipdb.set_trace()
        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean() + self.diff_weight * diff_loss 



    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def full_sort_predict(self, interaction):
#         ipdb.set_trace()
        user = interaction[0]  # 4096

        u_online, i_online = self.forward(self.norm_adj, self.sample_x)
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)

        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))  # 4096x64  64x7050
        return score_mat_ui
    
    def sample(self):
#         ipdb.set_trace()
#         user = interaction[0]  # 4096
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)

        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        predicted_x = self.diff.sample(self.model, self.item_id_embedding.weight, t_feat_online, v_feat_online)
#         ipdb.set_trace()
        self.sample_x = predicted_x
 

