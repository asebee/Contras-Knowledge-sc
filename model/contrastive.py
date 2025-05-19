import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
sys.path.append("../model/")  # 添加模型所在的路径
from load import *

#针对zheng68k
class ContrastiveLearningModel(nn.Module):

    def __init__(self, ckpt_path, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        #冻结位置和嵌入参数
        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        #只微调倒数encoder第二层参数
        # for na, param in self.encoder.named_parameters():
        #     param.requires_grad = False
        # for na, param in self.encoder.transformer_encoder[-2].named_parameters():
        #     print('self.encoder.transformer_encoder ', na, ' have grad')
        #     param.requires_grad = True

        self.fc1 = nn.Sequential(
            nn.Linear(model_config['encoder']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 11)  # 11个类别
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config

    def forward(self, sample_list, *args, **kwargs):
        x = sample_list['x']  # (B, L)
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x, x_padding)

        # mlp
        # logits, _ = torch.max(logits, dim=1)  # b,dim

        # logits = self.norm(logits)
        # logits = self.fc1(logits)

        return logits

    def compute_contrastive_loss(self, embeddings, labels, temperature=0.5):
        #构建一个标签掩码，用于标识哪些样本是正例
        #mask[i, j] 为 1 表示样本 i 和样本 j 具有相同的标签，为 0 表示它们具有不同的标签。
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # 除去自身相似度
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # 计算对比学习损失
        mask = mask * (1 - torch.eye(mask.size(0), device=mask.device))
        exp_logits = torch.exp(logits / temperature)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

if __name__ == '__main__':

    model = ContrastiveLearningModel(ckpt_path="path/to/checkpoint")
    model.build()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 输入数据和标签，维度为(52754, 19264)
    data=np.load("data_train_count.npy")
    labels=np.load("zheng-train-label.npy")
    data_tensor = torch.tensor(data).cuda()
    labels_tensor = torch.tensor(labels).cuda()
    sample_list = {'x': data_tensor, 'targets': labels_tensor}

    # sample_list = {'x': torch.zeros([8,18264]).cuda(),'targets':torch.rand(8,1).cuda()}
    # sample_list['x'][:,:100]=1
    # x_data = torch.randint(0, 100, (1000, 128))  # 输入数据
    # labels = torch.randint(0, 10, (1000,))  # 标签

    # 创建数据集和数据加载器
    dataset = TensorDataset(sample_list)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练过程
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            sample_list = {'x': batch[0], 'targets': batch[1]}
            
            # 前向传播
            logits = model(sample_list)
            
            # 计算对比学习损失
            embeddings = logits
            labels = sample_list['targets']
            contrastive_loss = model.compute_contrastive_loss(embeddings, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            contrastive_loss.backward()
            optimizer.step()
            
            total_loss += contrastive_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch%10==0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    #保存模型的Encoder部分
    encoder_state_dict = model.encoder.state_dict()
    torch.save(encoder_state_dict, 'encoder_constractive_with_label.ckpt')