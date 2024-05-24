import argparse
import configparser
import os
import pickle as pkl
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def data_param_prepare(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}
    #############################################################################################
    # Model
    embedding_dim = config.getint('Model', 'embedding_dim')  # 设置特征维度
    params['embedding_dim'] = embedding_dim

    n_layers = config.getint('Model', 'n_layers')
    params['n_layers'] = n_layers

    model_save_path = config['Model']['model_save_path']  # 设置模型保存路径
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')  # 设置epoch轮数
    params['max_epoch'] = max_epoch

    model_type = config['Model']['model_type']
    params['model_type'] = model_type
    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')  # 是否启动tensorboard

    ############################################################################################
    # Training
    dataset = config['Training']['dataset']  # 设置数据集
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']  # 训练集路径
    gpu = config['Training']['gpu']  # 设置gpu id
    params['gpu'] = gpu
    device = torch.device('cuda:' + params['gpu'] if torch.cuda.is_available() else "cpu")  # 设置tensor分配设备
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')  # 设置学习率
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')  # 设置batch-size
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')  # 设置早停轮数
    params['early_stop_epoch'] = early_stop_epoch

    negative_num = config.getint('Training', 'negative_num')  # 训练负例数
    params['negative_num'] = negative_num

    reg_weight = config.getfloat('Training', 'reg_weight')
    params['reg_weight'] = reg_weight

    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')  # 是否筛选正例
    params['sampling_sift_pos'] = sampling_sift_pos

    ##############################################################################################
    # Test
    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size  # 测试batch-size
    topk = config.getint('Testing', 'topk')  # top-k
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']  # 测试集路径
    ##############################################################################################
    # Datatset processing
    train_data, test_data, hyper_graph, user_num, item_num = load_data(train_file_path,
                                                                       test_file_path)  # 加载训练交互、测试交互、评分矩阵、用户总数、物品总数、限制矩阵
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                   num_workers=5)  # 将所有训练数据分为n_batch个子部分
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False,
                                  num_workers=5)  # 将所有测试数据分为n_batch个子部分
    # (数据,batch-size,是否打乱顺序,线程数)
    params['user_num'] = user_num  # 用户总数
    params['item_num'] = item_num  # 物品总数
    ####################################################################################################
    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:  # 训练集u-i对
        mask[u][i] = -np.inf
        interacted_items[u].append(i)  # 每一个用户的训练交互物品列表

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:  # 测试集u-i对
        test_ground_truth_list[u].append(i)  # 每一个用户的测试交互物品列表

    hypergraph = sparse_mx_to_torch_sparse_tensor(hyper_graph, device)

    return params, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, hypergraph


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def load_pickle(name):
    with open(name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


def split_data_randomly(user_records, test_ratio, seed):
    train_dict = {}
    test_dict = {}
    for user_id, item_list in enumerate(user_records):
        tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

        train_sample = []
        for place in item_list:
            if place not in tmp_test_sample:
                train_sample.append(place)

        test_sample = []
        for place in tmp_test_sample:
            test_sample.append(place)

        train_dict[user_id] = train_sample
        test_dict[user_id] = test_sample
    return train_dict, test_dict

def generate_adj(n_users, n_items, train_dict):
    ########################加载交互矩阵############################
    user_item = np.zeros((n_users, n_items)).astype(int)    # 生成一个u×i的全零矩阵

    for i, v in train_dict.items():
        user_item[i][v] = 1
    start = time.time()
    print('generating adj csr... ')
    coo_user_item = sp.coo_matrix(user_item).tocsr().astype(np.float32)     # 生成稀疏矩阵，转换成float32格式
    norm_coo_user_item = normalize_double(coo_user_item)    # 逐行求和，每个元素除以行和

    start = time.time()
    #########################超图稀疏矩阵构建###############################
    norm_user_item = np.array(norm_coo_user_item.todense())
    norm_item_user = norm_user_item.T
    norm_user_user = np.matmul(norm_user_item, norm_item_user)
    norm_item_item = np.matmul(norm_item_user, norm_user_item)

    # ------------------------gcn------------------------
    # norm_user_user = np.zeros(np.shape(norm_user_user))
    # norm_item_item = np.zeros(np.shape(norm_item_item))
    # ---------------------------------------------------

    # -----------------------hgnn------------------------
    # norm_user_item = np.zeros(np.shape(norm_user_item))
    # norm_item_user = np.zeros(np.shape(norm_item_user))
    # ---------------------------------------------------

    user = np.concatenate((norm_user_user, norm_user_item), axis=1)
    item = np.concatenate((norm_item_user, norm_item_item), axis=1)
    adj = np.concatenate((user, item), axis=0)  # 拼接成超图邻接矩阵(n+m)×(n+m)
    adj_csr = sp.coo_matrix(adj).tocsr().astype(np.float32) # coo→csr，生成压缩的稀疏矩阵

    #################### 加自连接 α=1 ####################################
    adj_dok = adj_csr.todok()
    adj_dok = adj_dok + sp.eye(adj_dok.shape[0])    # sp.eye():构建单位矩阵
    adj_csr = adj_dok.tocoo().astype(np.float32)

    #################################################################
    print('time elapsed: {:.3f}'.format(time.time() - start))
    return adj_csr


def normalize_double(mx):
    """Row-normalize sparse matrix."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    rowsum = np.array(mx.sum(1))
    r_inv_row = np.power(rowsum, -0.5).flatten()
    r_inv_row[np.isinf(r_inv_row)] = 0.

    colsum = np.array(mx.sum(0))
    r_inv_col = np.power(colsum, -0.5).flatten()
    r_inv_col[np.isinf(r_inv_col)] = 0.

    r_mat_inv_row = sp.diags(r_inv_row)     # 从对角线构造一个稀疏矩阵
    r_mat_inv_col = sp.diags(r_inv_col)

    mx = r_mat_inv_row.dot(mx)  # 点乘
    mx = mx.dot(r_mat_inv_col)
    return mx

def load_data(train_file, test_file):
    # pkl_path = 'D:\gjj\LDHC-pytorch\LHDC-main\data\dataset_cd\interaction.pkl'
    # pkl_path = 'C:/Users/Clara/Desktop/graduate/code/LDHC-pytorch-txt/LHDC-main/data/dataset_cd/interaction.pkl'
    # user_item_list = load_pickle(pkl_path)
    # train_items, test_set = split_data_randomly(user_item_list, test_ratio=0.2, seed=1234)
    # n_user, m_item = len(user_item_list), max([max(x) for x in user_item_list]) + 1
    n_user, m_item = 0, 0
    train_items, test_set = {}, {}
    with open(train_file) as f:
        for l1 in f.readlines():
            if len(l1) > 0:
                l1 = l1.strip('\n').split(' ')
                items = [int(i) for i in l1]
                uid, train_item = items[0], items[1:]
                train_items[uid] = train_item
                n_user = max(n_user, uid)
                # edit
                m_item = max(m_item, max(train_item))
    n_user += 1
    with open(test_file) as f:
        for l2 in f.readlines():
            if len(l2) > 0:
                l2 = l2.strip('\n').split(' ')
                items = [int(i) for i in l2]
                uid, test_item = items[0], items[1:]
                test_set[uid] = test_item
                # m_item = max(m_item, max(items))
    m_item += 1
    hypergraph = generate_adj(n_user, m_item, train_items)

    #############################################################################
    # n_train, n_test = 0, 0
    #
    # for uid in range(n_user):
    #     lenth_train = len(train_items[uid])
    #     lenth_test = len(test_set[uid])
    #     n_train = n_train + lenth_train
    #     n_test = n_test + lenth_test

    trainItem, trainUser = [], []
    for uid in train_items.keys():
        items = train_items[uid]
        trainUser.extend([uid] * len(items))  # uid[0, 0, 0, 0, 1, 1]
        trainItem.extend(items)  # iid[13, 14, 15, 16, 3, 4]

    testItem, testUser = [], []
    for uid in test_set.keys():
        items = test_set[uid]
        testUser.extend([uid] * len(items))
        testItem.extend(items)

    train_data = []  # 储存训练交互的坐标
    test_data = []  # 储存测试交互的坐标

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])

    return train_data, test_data, hypergraph, n_user, m_item


'''
Useful functions
'''


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
    neg_candidates = np.arange(item_num)  # 起点0 终点n_items-1 步长1

    if sampling_sift_pos:
        neg_items = []
        for u in pos_train_data[0]:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)

    return pos_train_data[0], pos_train_data[1], neg_items  # users, pos_items, neg_items


'''
Model Definition
'''


class LDHC(nn.Module):
    def __init__(self, params, hypergraph):
        super(LDHC, self).__init__()

        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.n_layers = params['n_layers']

        self.reg_weight = params['reg_weight']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.hypergraph = hypergraph

        self.initial_weights()

    def initial_weights(self):
        nn.init.xavier_normal_(self.user_embeds.weight)
        nn.init.xavier_normal_(self.item_embeds.weight)

    def _get_ego_embeddings(self):
        # concat of user embeddings and item embeddings
        user_emb = self.user_embeds.weight
        item_emb = self.item_embeds.weight
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        return ego_embeddings

    def forward(self):  # 修改
        ego_embeddings = self._get_ego_embeddings()
        all_embeddings = [ego_embeddings]   # list
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.hypergraph, ego_embeddings)   # HGNN, equation12
            temp_embeddings = ego_embeddings
            all_embeddings += [temp_embeddings] # 将每层卷积结果存入all_embeddings列表

        all_embeddings = torch.cat(all_embeddings, dim=1)   # 横向拼接每层embedding

        u_g_embeddings = all_embeddings[:self.user_num, :]  #
        i_g_embeddings = all_embeddings[self.user_num:, :]
        # u_g_embeddings = ego_embeddings[:self.user_num]
        # i_g_embeddings = ego_embeddings[self.user_num:]
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, users, pos_items, neg_items):
        # model forward
        u_g_embeddings, i_g_embeddings = self.forward()
        ###########################################################################################
        # regularization loss
        u_ego_embeddings = self.user_embeds(users)
        pos_ego_embeddings = self.item_embeds(pos_items)
        neg_ego_embeddings = self.item_embeds(neg_items)

        reg_loss = (torch.norm(u_ego_embeddings) ** 2
                    + torch.norm(pos_ego_embeddings) ** 2
                    + torch.norm(neg_ego_embeddings) ** 2) / 2

        reg_loss = self.reg_weight * reg_loss
        ##########################################################################################
        # bpr loss
        # users.tolist()
        # pos_items.tolist()
        # neg_items.tolist()

        u_g_embeddings_bpr = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :].squeeze(1)

        pos_scores = torch.mul(u_g_embeddings_bpr, pos_i_g_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_g_embeddings_bpr, neg_i_g_embeddings).sum(dim=1)
        bpr_loss = self.create_bpr_loss(pos_scores, neg_scores)
        ############################################################################################
        # total loss
        loss = bpr_loss + reg_loss

        return loss, bpr_loss, reg_loss

    def create_bpr_loss(self, pos_scores, neg_scores):
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        bpr_loss = -1 * torch.mean(maxi)

        return bpr_loss

    def test_foward(self, users):
        user_embeddings, item_embeddings = self.forward()
        users = users.tolist()
        user_embeds = user_embeddings[users, :]  # batch users

        return user_embeds.mm(item_embeddings.t())

    def get_device(self):
        return self.user_embeds.weight.device


'''
Train
'''


########################### TRAINING #####################################
def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params):
    device = params['device']
    best_epoch, best_recall, best_ndcg, best_pre, best_F1 = 0, 0, 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))

    if params['enable_tensorboard']:
        writer = SummaryWriter()  # 建立tensorboard进程

    for epoch in range(params['max_epoch']):
        model.train()  # 模型使用BN和dropout
        start_time = time.time()  # 开始时间

        for batch, x in enumerate(tqdm(train_loader)):  # x: tensor:[users, pos_items]
            users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], interacted_items,
                                                   params['sampling_sift_pos'])  # 为每一个用户采集正例和负例(1:N)
            users = users.type(torch.long).to(device)  # 将采样后的用户、物品正负例放入GPU中
            pos_items = pos_items.type(torch.long).to(device)
            neg_items = neg_items.type(torch.long).to(device)

            model.zero_grad()  # 清空梯度
            loss, bpr_loss, reg_loss = model.calculate_loss(users, pos_items, neg_items)  # 前向传播 计算loss  执行forward函数
            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()  # 反向传播 计算梯度
            optimizer.step()  # 更新参数

        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))  # 格式化时间 gmt格林威治时间
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)  # tensoraboard生成loss曲线
        print("Loss = {:.5f} \t bpr_loss = {:.5f} \t reg_loss = {:.5f}".format(loss.item(), bpr_loss, reg_loss))

        need_test = True
        if epoch < 2 and epoch % 1 != 0:  # 前50轮 每5轮test一次
            need_test = False

        if need_test:  # 开始时间
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, params['topk'],
                                                     params['user_num'])
            if params['enable_tensorboard']:
                writer.add_scalar('Results/recall@20', Recall, epoch)
                writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print(
                "Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(),
                                                                                                            F1_score,
                                                                                                            Precision,
                                                                                                            Recall,
                                                                                                            NDCG))

            if Recall > best_recall:
                best_recall, best_ndcg, best_pre, best_F1, best_epoch = Recall, NDCG, Precision, F1_score, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])

            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True

        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}, best precision = {}, best F1-score = {}'.format(
                best_epoch, best_recall, best_ndcg, best_pre, best_F1))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    writer.flush()  # 刷新缓冲区

    print('Training end!')


# The below 7 functions (hit, ndcg, RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) follow this license.
# MIT License

# Copyright (c) 2020 Xiang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################### TESTING #####################################
'''
Test and metrics
'''


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():  # test不计算梯度
        model.eval()  # 模型不使用BN和dropout
        for idx, batch_users in enumerate(tqdm(test_loader)):  # 每个batch中的用户

            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users)  # 每个batch user对所有物品的评分
            rating = rating.cpu()  # 评分tensor放入cpu
            batch_users = batch_users.cpu()
            rating += mask[batch_users]  # 将训练集正例得分mask

            _, rating_K = torch.topk(rating, k=topk)  # 返回每一用户得分topk的索引(物品id)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./amazoncds_config.ini', type=str, help='config file path')
    # parser.add_argument('--config_file', default='./mfbpr_citeulike_config.ini', type=str, help='config file path')
    args = parser.parse_args()

    print('###################### LDHC ######################')

    print('1. Loading Configuration...')
    params, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, hypergraph = data_param_prepare(
        args.config_file)

    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    ldhc = LDHC(params, hypergraph)  # 初始化模型

    ldhc = ldhc.to(params['device'])

    optimizer = torch.optim.Adam(ldhc.parameters(), lr=params['lr'])

    train(ldhc, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    print('END')
