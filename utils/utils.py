import torch
from typing import Union
import torch.nn as nn
from torch.nn import functional as F
from utils.get_args import threshold
from sklearn.metrics.pairwise import euclidean_distances
from utils.get_args import get_args
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

class HyP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.args = get_args()
        torch.manual_seed(self.args.hypseed)

        # self.proxies = nn.Parameter(torch.zeros(self.args.numclass, self.args.output_dim).to(1))
        #Initialization
        self.proxies = torch.nn.Parameter(torch.randn(self.args.numclass, self.args.output_dim).to(self.args.device))
        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')
        self.proxies.data = torch.tanh(self.proxies.data)
        #self.margin=3

        self.maxfunc = torch.nn.ReLU()

        #修改
        # labels = scio.loadmat("/home/cgh/DSPH-main/dataset/flickr25k/label.mat")["category"]
        # self.labels = torch.tensor(labels, dtype=torch.float32).to(self.args.device)
        # #row_count = self.labels.size(0)
        # N = self.labels.T.mm(self.labels)
        # self.max_off_diag = N.masked_fill(torch.eye(N.size(0)).to(N.device).bool(), float('-inf')).max()
        # self.M = N *1.3/self.max_off_diag
#修改11
    def compute_new_loss(self, proxies,m):
        # 计算代理向量之间的内积
        inner_product_matrix = F.normalize(proxies, p = 2, dim = 1).mm(F.normalize(proxies, p = 2, dim = 1).T)  # 形状 (c, c)

        # 逐元素比较并求和（排除对角线元素）
        max_matrix = torch.max(m, inner_product_matrix)
        sum_result = max_matrix.sum() - max_matrix.diag().sum()

        return sum_result

    def forward(self, x=None, y=None, label=None):
        P_one_hot = label

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos)

        cos_t = F.normalize(y, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos_t = 1 - cos_t
        neg_t = F.relu(cos_t)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())

        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num

        pos_term_t = torch.where(P_one_hot  ==  1, pos_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / P_num
        neg_term_t = torch.where(P_one_hot  ==  0, neg_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / N_num

        label = label.float()
        L = F.normalize(label, p=2, dim=1).mm(F.normalize(label, p=2, dim=1).T)
        batch_size = L.size(0)
        # nan_tensor = torch.tensor([float('nan')], dtype=torch.float32).to(self.args.device)
        # 计算每一行中对应 P_one_hot == 1 的值的方差
        pos_variances = torch.var(torch.where(P_one_hot == 1, pos, torch.zeros_like(pos)), dim=1)
        pos_t_variances = torch.var(torch.where(P_one_hot == 1, pos_t, torch.zeros_like(pos_t)), dim=1)

        # 计算方差损失
        variance_loss = pos_variances.mean() + pos_t_variances.mean()
        #使用 torch.nanvar 忽略 nan 值
        # pos_valid = torch.where(P_one_hot == 1, pos, torch.tensor(float('nan')).to(self.args.device))
        # pos_t_valid = torch.where(P_one_hot == 1, pos_t, torch.tensor(float('nan')).to(self.args.device))
        #
        # pos_variances = []
        # pos_t_variances = []
        #
        # for i in range(pos_valid.size(0)):
        #     valid_pos = pos_valid[i][~torch.isnan(pos_valid[i])]
        #     valid_pos_t = pos_t_valid[i][~torch.isnan(pos_t_valid[i])]
        #     if valid_pos.numel() > 0:
        #         pos_variances.append(torch.var(valid_pos, unbiased=False))
        #     if valid_pos_t.numel() > 0:
        #         pos_t_variances.append(torch.var(valid_pos_t, unbiased=False))
        #
        # pos_variances = torch.tensor(pos_variances).to(self.args.device)
        # pos_t_variances = torch.tensor(pos_t_variances).to(self.args.device)
        #
        # total_variance_loss = (pos_variances.sum() + pos_t_variances.sum())/batch_size


        # S_pq=calculate_dissimilarity_matrix_tensor(label, self.args.numclass)
        # L=replace_similarity_with_dissimilarity_tensor(L, S_pq)
        # L = L.float()

        # S = (L > 0).float()
        # L=(L+S)/2

        x_sim = F.normalize(x, p=2, dim=1).mm(F.normalize(x, p=2, dim=1).T)
        y_sim = F.normalize(y, p=2, dim=1).mm(F.normalize(y, p=2, dim=1).T)
        xy_sim = F.normalize(x, p=2, dim=1).mm(F.normalize(y, p=2, dim=1).T)
        yx_sim = F.normalize(y, p=2, dim=1).mm(F.normalize(x, p=2, dim=1).T)
        pos =  F.relu(L-x_sim)
        pos_y =  F.relu(L-y_sim)
        pos_xy =  F.relu(L-xy_sim)
        pos_yx =  F.relu(L - yx_sim)
        pos_term_x=torch.where(L > 0, pos, torch.zeros_like(pos)).sum() / len((L > 0).nonzero())
        pos_term_y = torch.where(L > 0, pos_y, torch.zeros_like(pos_y)).sum() / len((L > 0).nonzero())
        pos_term_xy = torch.where(L > 0, pos_xy, torch.zeros_like(pos_xy)).sum() / len((L > 0).nonzero())
        #pos_term_yx = torch.where(L > 0, pos_yx, torch.zeros_like(pos_yx)).sum() / len((L > 0).nonzero())
        p2p=(pos_term_x + pos_term_y + pos_term_xy)






        index = label.sum(dim = 1) > 1
        label_ = label[index].float()

        x_ = x[index]
        t_ = y[index]

        cos_sim = label_.mm(label_.T)

        if len((cos_sim == 0).nonzero()) == 0:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0
            #reg_term_tx = 0

        else:
            x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
            t_sim = F.normalize(t_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)
            xt_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)
            #tx_sim = F.normalize(t_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)


            neg =  F.relu(x_sim)
            neg_t =  F.relu(t_sim)
            neg_xt =  F.relu(xt_sim)
            #neg_tx = self.args.alpha * F.relu(tx_sim -threshold)


            reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
            reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len((cos_sim == 0).nonzero())
            reg_term_xt = torch.where(cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)).sum() / len((cos_sim == 0).nonzero())
            #reg_term_tx = torch.where(cos_sim == 0, neg_tx, torch.zeros_like(tx_sim)).sum() / len((cos_sim == 0).nonzero())
        n2n=reg_term + reg_term_t + reg_term_xt
        #new_loss = self.compute_new_loss(self.proxies, self.M)/(self.args.numclass*(self.args.numclass-1))

        return  pos_term + pos_term_t+neg_term +neg_term_t+0.8*n2n+0.05*p2p+variance_loss

        #+loss1+loss2

#+ reg_term + reg_term_t + reg_term_xt + neg_term + neg_term_t
#pos_term + pos_term_t + neg_term + neg_term_t + 0.8 * new_loss


    def compute_metrics(x):
    # 取复值的原因在于cosine的值越大说明越相似，但是需要取的是前N个值，所以取符号变为增函数s
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
        d = d[:, np.newaxis]
        ind = sx - d
        ind = np.where(ind == 0)
        ind = ind[1]
        metrics = {}
        metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
        metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
        metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
        metrics['MR'] = np.median(ind) + 1
        metrics["MedianR"] = metrics['MR']
        metrics["MeanR"] = np.mean(ind) + 1
        metrics["cols"] = [int(i) for i in list(ind)]
        return metrics

    def B_loss(self, x=None, y=None, label=None):
        # 定义你的第二种损失函数
        label = label.float()

        L = F.normalize(label, p=2, dim=1).mm(F.normalize(label, p=2, dim=1).T)
        # S_pq=calculate_dissimilarity_matrix_tensor(label, self.args.numclass)
        # L=replace_similarity_with_dissimilarity_tensor(L, S_pq)
        # L = L.float()

        # S = (L > 0).float()
        # L=(L+S)/2
        batch_size = L.size(0)
        k = torch.tensor(self.args.output_dim, dtype=torch.float32)
        A = torch.tensor(1, dtype=torch.float32)
        B = torch.tensor(1, dtype=torch.float32)
        thresh = (1 - L) * k / 2
        width = 3
        up_thresh = thresh
        low_thresh = thresh - width
        low_thresh[low_thresh <= 0] = 0
        low_thresh[L == 0] = self.args.output_dim / 2

        # low_flag = torch.ones(batch_size, batch_size).cuda(self.args.device)
        # up_flag = torch.ones(batch_size, batch_size).cuda(self.args.device)
        # low_flag[L == 1] = 0
        # low_flag[L == 0] = A
        # up_flag[L == 0] = 0
        # up_flag[L == 1] = B

        BI_BI = (self.args.output_dim - x .mm (x.t())) / 2
        BT_BT = (self.args.output_dim - y .mm (y.t())) / 2
        BI_BT = (self.args.output_dim - x .mm (y.t())) / 2
        BT_BI = (self.args.output_dim - y .mm (x.t())) / 2
        #
        # # # lower bound
        loss1 = (torch.norm(self.maxfunc(low_thresh - BI_BI)) \
                 + torch.norm(self.maxfunc(low_thresh - BT_BT)) \
                 + torch.norm(self.maxfunc(low_thresh - BT_BI)) \
                 + torch.norm(self.maxfunc(low_thresh - BI_BT))) / (batch_size * batch_size)

        # upper bound
        loss2 = (torch.norm(self.maxfunc(BI_BI - up_thresh)) \
                 + torch.norm(self.maxfunc(BT_BT - up_thresh) ) \
                 + torch.norm(self.maxfunc(BT_BI - up_thresh) ) \
                 + torch.norm(self.maxfunc(BI_BT - up_thresh) )) / (batch_size * batch_size)

        return loss1+loss2


def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def euclidean_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        similarity = torch.cdist(a, b, p=2.0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        similarity = euclidean_distances(a, b)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    return similarity


def euclidean_dist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1: a tensor with shape (a, c)
    :param tensor2: a tensor with shape (b, c)
    :return: the euclidean distance matrix which each point is the distance between a row in tensor1 and a row in tensor2.
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist


def cosine_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / a.norm(dim=-1, keepdim=True) if len(torch.where(a != 0)[0]) > 0 else a
        b = b / b.norm(dim=-1, keepdim=True) if len(torch.where(b != 0)[0]) > 0 else b
        return torch.matmul(a, b.t())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) if len(np.where(a != 0)[0]) > 0 else a
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) if len(np.where(b != 0)[0]) > 0 else b
        return np.matmul(a, b.T)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    # print("query num:", num_query)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.to(rank)
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calcHammingDist(B1, B2):

    if len(B1.shape) < 2:
        B1.view(1, -1)
    if len(B2.shape) < 2:
        B2.view(1, -1)
    q = B2.shape[1]
    if isinstance(B1, torch.Tensor):
        distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    elif isinstance(B1, np.ndarray):
        distH = 0.5 * (q - np.matmul(B1, B2.transpose()))
    else:
        raise ValueError("B1, B2 must in [torch.Tensor, np.ndarray]")
    return distH


def xor_dissimilarity_tensor(y_p, y_q, K):
    """
    计算两个张量之间的XOR不相似度。

    参数:
    y_p -- 第一个输入张量
    y_q -- 第二个输入张量
    K -- 类别数量

    返回:
    S_pq -- 计算得到的不相似度得分
    """
    # 确保输入是整数类型
    y_p = y_p.long()
    y_q = y_q.long()

    # 计算XOR结果
    xor_result = torch.logical_xor(y_p, y_q)

    # 计算非重叠的数量
    non_overlap = torch.sum(xor_result)

    # 计算不相似度得分
    S_pq = -non_overlap.float() / K

    return S_pq


def calculate_dissimilarity_matrix_tensor(labels, K):
    """
    计算不相似度矩阵。

    参数:
    labels -- 标签张量
    K -- 类别数量

    返回:
    S_pq -- 不相似度矩阵
    """
    num_instances = labels.shape[0]
    S_pq = torch.zeros((num_instances, num_instances), device=labels.device)

    for i in range(num_instances):
        for j in range(num_instances):
            S_pq[i, j] = xor_dissimilarity_tensor(labels[i], labels[j], K)

    return S_pq


def replace_similarity_with_dissimilarity_tensor(L, S_pq):
    # 创建一个布尔掩码，指示L中哪些位置为0
    mask = (L == 0)

    # 使用torch.where来替换L中的值
    L = torch.where(mask, S_pq, L)

    return L



def pr_curve(qF, rF, qL, rL, what=0, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    if what == 0:
        Rank = np.argsort(cdist(qF, rF, 'cosine'))
    else:
        Rank = np.argsort(cdist(qF, rF, 'hamming'))

    P, R = [], []
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K

        P.append(np.mean(p))
        R.append(np.mean(r))

    # 画 P-R 曲线
    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P)  # 第一个是 x，第二个是 y
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.legend()
    plt.show()
