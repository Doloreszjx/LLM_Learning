#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# Define: Experts
class Experts(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(Experts, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
    def forward(self, x):
        return self.ffn(x)

# MoE -- Top1
class MoETop1(nn.Module):
    def __init__(self, d_model, d_hidden, num_experts, capacity_factor=1.25, aux_loss_coef=1e-2):
        super(MoETop1, self).__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        # router: 根据token embedding -》 loggias
        self.router = nn.Linear(d_model, num_experts)
        # 专家序列
        self.experts = nn.ModuleList([Experts(d_model, d_hidden) for _ in range(num_experts)])

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model], aux_loss, used_counts
        :param x:
        :return:
        """
        batch_size, seq_len, d_model = x.shape
        tokens = batch_size * seq_len
        # 目的是 “消除批量和序列的维度区分”，让每个 token 成为独立的 “单样本”
        # 比如（512， 768）-- 512 行对应 512 个 token，每行 768 维是该 token 的特征
        x_flat = x.reshape(tokens, d_model)

        # 路由器 gating
        # (tokens, E)
        logits = self.router(x_flat)
        # soft prob
        gates = F.softmax(logits, dim=-1)
        # (tokens, )
        top_vals, top_idx = gates.max(dim=-1)

        # 负载均衡：避免“偏心”只激活某几个专家，辅助loss
        # 门控网络输出的专家激活概率，按维度0（所有token取平均
        importance =gates.mean(0)
        # 用 one-hot 编码转为 “token - 专家” 的 0/1 矩阵（选中为 1，未选中为 0），再转浮点型方便计算
        one_hot = F.one_hot(top_idx, self.num_experts).float()
        load = one_hot.mean(0)
        # 得到每个专家的 “平均负载率”，即被多少比例的 token 实际选中，反映专家的真实使用频率。
        aux_loss = (importance * load).sum() * self.num_experts * self.aux_loss_coef

        # 容量限制
        capacity = int(self.capacity_factor * tokens / self.num_experts)

        # 输出缓冲
        y_flat = torch.zeros_like(x_flat)
        used_counts = torch.zeros(self.num_experts, dtype=torch.int)

        for expert in range(self.num_experts):
            idx_e = (top_idx == expert).nonzero(as_tuple=True)[0]
            if idx_e.numel() > 0:
                idx_e = idx_e[:capacity]
                used_counts[expert] = idx_e.numel()
                inp = x_flat[idx_e]
                output = self.experts[expert](inp) * top_vals[idx_e].unsqueeze(-1)
                y_flat.index_add(0, idx_e, output)

        return y_flat.view(batch_size, seq_len, d_model), aux_loss, used_counts

# ===== 测试 Demo =====
if __name__ == "__main__":
    torch.manual_seed(42)

    d_model = 16
    d_hidden = 32
    num_experts = 4
    batch_size = 8
    seq_len = 10

    model = MoETop1(d_model, d_hidden, num_experts)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 简单训练 loop
    for step in range(5):
        x = torch.randn(batch_size, seq_len, d_model)
        y_true = torch.randn(batch_size, seq_len, d_model)

        y_pred, aux_loss, used_counts = model(x)
        loss = F.mse_loss(y_pred, y_true) + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss={loss.item():.4f}, aux={aux_loss.item():.4f}, used={used_counts.tolist()}")

    # 单次推理
    x = torch.randn(2, 5, d_model)
    y, aux_loss, used_counts = model(x)
    print("Inference shape:", y.shape)
    print("Experts used:", used_counts.tolist())