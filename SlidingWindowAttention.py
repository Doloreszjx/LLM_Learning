#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    """
       初始化滑动窗口注意力
       :param d_model: 输入token的维度（如Q/K/V的维度，需为偶数，方便Scaled Dot-Product）
       :param window_size: 滑动窗口大小（每个token能关注的"左右+自身"的总数量，建议为奇数）
    """
    def __init__(self, d_model, window_size=3):
        super(SlidingWindowAttention, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.half_window = window_size // 2
        # 1. 生成Q/K/V的线性层（将输入token映射到Q、K、V空间）
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # 2. 缩放因子（Scaled Dot-Product的核心，避免梯度消失）
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))  # sqrt(d_model)

    def forward(self, x):
        """
            前向传播：计算滑动窗口注意力
            :param x: 输入序列，形状为 [batch_size, seq_len, d_model]
                      batch_size：批量大小（一次处理多少个序列）
                      seq_len：序列长度（每个序列有多少个token）
                      d_model：每个token的维度
                      x 是输入序列，形状是 [batch_size, seq_len, d_model]（比如 [1, 5, 4]，代表 1 个 batch、5 个 token、每个 token 维度 4）。
            :return: 注意力输出，形状与输入x一致 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        # 步骤1：生成Q、K、V（每个token映射到对应的Query、Key、Value）
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        # 步骤2：生成"滑动窗口掩码"——限制每个token只能关注窗口内的K
        # 掩码形状：[seq_len, seq_len]，值为0表示"允许关注"，值为-1e9表示"禁止关注"（Softmax后接近0）
        mask = torch.ones(seq_len, seq_len, device=x.device) * (-1e9)  # 先初始化全为"禁止"

        # 遍历每个token的位置i，为其划定窗口范围 [i - half_window, i + half_window]
        for i in range(seq_len):
            # 左边界不小于0
            left = max(0, i - self.half_window)
            # 右边界不超过每个token的序列长度
            right = min(seq_len - 1, i + self.half_window)
            # Python切片是左闭右开
            mask[i, left:right+1] = 0
        # 步骤3：计算Q与K的相似度（Scaled Dot-Product核心）
        # Q形状 [batch_size, seq_len, d_model]，K转置后 [batch_size, d_model, seq_len]
        # 矩阵乘法后得到注意力分数：[batch_size, seq_len, seq_len]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # 步骤4：应用滑动窗口掩码——将窗口外的分数设为-1e9（Softmax后权重接近0）
        attn_scores = attn_scores + mask.unsqueeze(0)  # 给掩码加batch维度，匹配attn_scores形状

        # 步骤5：Softmax归一化——将分数转为"注意力权重"（每个token对窗口内token的关注程度，总和为1）
        # dim=-1 就是 “对注意力分数矩阵的最后一个维度（即‘被关注 token’的维度）做 Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 步骤6：加权求和Value——用注意力权重对V加权，得到最终注意力输出
        output = torch.bmm(attn_weights, V)  # [batch_size, seq_len, d_model]

        return output, attn_weights  # 返回输出和注意力权重（可用于可视化关注情况）


# ------------------- 测试代码：用简单案例验证 -------------------
if __name__ == "__main__":
    # 1. 构造输入：batch_size=1（1个序列），seq_len=5（5个token），d_model=4（每个token维度为4）
    x = torch.randn(1, 5, 4)  # 随机生成输入（模拟5个token的序列）

    # 2. 初始化滑动窗口注意力：窗口大小=3（每个token看左右1个+自身）
    sw_attn = SlidingWindowAttention(d_model=4, window_size=3)

    # 3. 计算注意力输出和权重
    output, attn_weights = sw_attn(x)

    # 4. 打印结果，验证逻辑
    print("输入序列形状：", x.shape)  # 输出：torch.Size([1, 5, 4])
    print("注意力输出形状：", output.shape)  # 输出：torch.Size([1, 5, 4])（与输入一致）
    print("\n注意力权重矩阵（窗口大小=3，每行代表1个token的关注权重）：")
    print(attn_weights.squeeze(0))  # 去掉batch维度，打印5x5的权重矩阵