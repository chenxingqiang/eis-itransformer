import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEncoding


class EISModule(nn.Module):
    def __init__(
        self, d_model, nhead, num_layers, dim_feedforward, num_ecm_params, dropout=0.1
    ):
        super(EISModule, self).__init__()
        self.freq_encoder = nn.Linear(1, d_model)
        self.imp_encoder = nn.Linear(3, d_model)
        self.ecm_encoder = nn.Linear(num_ecm_params, d_model)  # 新增等效电路参数编码器
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, freq, imp, ecm_params):
        freq_embed = self.freq_encoder(freq)
        imp_embed = self.imp_encoder(imp)
        ecm_embed = self.ecm_encoder(ecm_params)  # 编码等效电路参数
        src = freq_embed + imp_embed + ecm_embed
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

    def calculate_loss(self, eis_data, targets):
        freq, imp, ecm_params = eis_data
        outputs = self.forward(freq, imp, ecm_params)

        # 计算 EIS 损失,例如使用均方误差损失
        eis_loss = nn.MSELoss()(outputs, targets)

        return eis_loss


import torch
from torch.utils.data import Dataset


class EISDataset(Dataset):
    def __init__(self, eis_data):
        self.eis_data = eis_data

    def __len__(self):
        return len(self.eis_data)

    def __getitem__(self, idx):
        # 从 EIS 数据中获取单个样本
        sample = self.eis_data.iloc[idx]

        # 提取频率、阻抗和等效电路参数
        freq = torch.tensor(sample["freq"], dtype=torch.float32)
        imp_real = torch.tensor(sample["imp_real"], dtype=torch.float32)
        imp_imag = torch.tensor(sample["imp_imag"], dtype=torch.float32)
        imp_mag = torch.tensor(sample["imp_mag"], dtype=torch.float32)
        imp_phase = torch.tensor(sample["imp_phase"], dtype=torch.float32)
        ecm_params = torch.tensor(
            [sample[f"ecm_param_{i}"] for i in range(self.num_ecm_params)],
            dtype=torch.float32,
        )

        # 创建阻抗张量
        imp = torch.stack([imp_real, imp_imag, imp_mag, imp_phase])

        # 返回频率、阻抗和等效电路参数
        return freq, imp, ecm_params
