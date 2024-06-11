import torch
import torch.nn.functional as F
'''
def arcface_loss(logits:torch.Tensor, labels:torch.Tensor, margin: float, s=64.0):
    # Ensure logits are normalized
    cos_theta = F.normalize(logits, dim=1)
    # Ensure label weights are normalized
    cos_theta = F.normalize(cos_theta, dim=1)
    # Get the target logits
    target_logits = cos_theta[torch.arange(len(logits)), labels]
    # Add the margin to the target logits
    target_logits_with_margin = torch.cos(torch.acos(target_logits) + margin)
    # Scale the logits
    scaled_logits = s * cos_theta
    scaled_target_logits = s * target_logits_with_margin
    # Replace the target logits with the modified target logits
    logits_with_margin = scaled_logits.clone()
    logits_with_margin[torch.arange(len(logits)), labels] = scaled_target_logits
    # Compute the loss
    loss = F.cross_entropy(logits_with_margin, labels)
    return loss
'''
# def arcface_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float, s=64.0):
#     # Ensure logits are normalized
#     cos_theta = F.normalize(logits, dim=1)
#     # Ensure label weights are normalized
#     cos_theta = F.normalize(cos_theta, dim=1)
    
#     # Map labels to the correct range
#     min_label = labels.min().item()
#     labels -= min_label
    
#     # Get the target logits
#     target_logits = cos_theta[torch.arange(len(logits)), labels]
#     # Add the margin to the target logits
#     target_logits_with_margin = torch.cos(torch.acos(target_logits) + margin)
#     # Scale the logits
#     scaled_logits = s * cos_theta
#     scaled_target_logits = s * target_logits_with_margin
#     # Replace the target logits with the modified target logits
#     logits_with_margin = scaled_logits.clone()
#     logits_with_margin[torch.arange(len(logits)), labels] = scaled_target_logits
#     # Compute the loss
#     loss = F.cross_entropy(logits_with_margin, labels)
#     return loss
import torch
import torch.nn.functional as F


def arcface_loss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float) -> torch.Tensor:
    
    # 步驟 1: 特徵歸一化 (Normalize embedding features)
    embedding = F.normalize(embedding, dim=1)
    
    # 步驟 2: 計算成對餘弦相似度矩陣 (Compute pairwise cosine similarity matrix)
    dist_mat = torch.matmul(embedding, embedding.t())
    
    N = dist_mat.size(0)
    # 創建正樣本和負樣本掩碼 (Create positive and negative masks)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # 掩蓋自身的分數 (Mask scores related to itself)
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    # 步驟 3: 正樣本和負樣本的 cos 類似度 (Positive and negative cosine similarity)
    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    # 步驟 4: 計算正樣本的角度 (Compute angles for positive pairs)
    angles = torch.acos(torch.clamp(s_p, -1.0 + 1e-7, 1.0 - 1e-7))
    
    # 步驟 5: 在角度上添加 margin (Add margin to angles)
    angles_with_margin = angles + margin * is_pos

    # 步驟 6: 將角度轉換回餘弦相似度 (Convert angles back to cosine similarity)
    cosine_with_margin = torch.cos(angles_with_margin)
    
    # 步驟 7: 計算加上 margin 的 logits (Compute logits with margin for positive pairs)
    logit_p = gamma * cosine_with_margin + (-99999999.) * (1 - is_pos)
    logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)

    # 步驟 8: 損失計算 (Loss calculation)
    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss

# loss = pairwise_arcface(embedding, targets, margin, gamma)
# print(loss)


# def arcface_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float, s=64.0):
#     # 確保 logits 是已歸一化的
#     cos_theta = F.normalize(logits, dim=1)
    
#     # 確保 label 的範圍在 logits 的範圍內
#     num_classes = labels.max().item() + 1
#     if num_classes > logits.size(1):
#         padding_size = num_classes - logits.size(1)
#         padding = torch.zeros(logits.size(0), padding_size, device=logits.device)
#         logits = torch.cat((logits, padding), dim=1)
#         cos_theta = torch.cat((cos_theta, padding), dim=1)
    
#     # 獲取目標 logits
#     target_logits = cos_theta[torch.arange(len(logits)), labels]
    
#     # 對目標 logits 添加 margin
#     target_logits_with_margin = torch.cos(torch.acos(target_logits) + margin)
    
#     # 縮放 logits
#     scaled_logits = s * cos_theta
#     scaled_target_logits = s * target_logits_with_margin
    
#     # 替換目標 logits
#     logits_with_margin = scaled_logits.clone()
#     logits_with_margin[torch.arange(len(logits)), labels] = scaled_target_logits
    
#     # 計算損失
#     loss = F.cross_entropy(logits_with_margin, labels)
#     return loss


# # Usage example
# logits = torch.randn(32, 10)  # Example logits (batch size 32, 10 classes)
# labels = torch.randint(0, 10, (32,))  # Example labels within the valid range

# loss = arcface_loss(logits, labels, margin=0.5)
# print(loss)

# logits = torch.randn(128, 2048)  # Example logits (batch size 32, 10 classes)
# labels = torch.randint(0, 10, (128,))  # Example labels
# margin = 0.5
# gamma = 0.1
# s = 28.0
# loss = arcface_loss(logits, labels, margin, gamma)
# print(loss)
# class ArcFaceLoss(torch.nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFaceLoss, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, logits, labels):
#         # Ensure logits are normalized
#         cos_theta = F.normalize(logits, dim=1)
#         # Ensure label weights are normalized
#         cos_theta = F.normalize(cos_theta, dim=1)
        
#         # Get the target logits
#         target_logits = cos_theta[torch.arange(len(logits)), labels]
        
#         # Add the margin to the target logits
#         target_logits_with_margin = torch.cos(torch.acos(target_logits) + self.m)
        
#         # Scale the logits
#         scaled_logits = self.s * cos_theta
#         scaled_target_logits = self.s * target_logits_with_margin
        
#         # Replace the target logits with the modified target logits
#         logits_with_margin = scaled_logits.clone()
#         logits_with_margin[torch.arange(len(logits)), labels] = scaled_target_logits
        
#         # Compute the loss
#         loss = F.cross_entropy(logits_with_margin, labels)
        
#         return loss


# def arcface_loss(embedding: torch.Tensor, targets: torch.Tensor, margin: float, gamma: float) -> torch.Tensor:
#     # Step 1: Normalize embedding features
#     embedding = F.normalize(embedding, dim=1)
    
#     # Step 2: Compute cosine similarity between embeddings and class centers
#     cosine_sim = torch.matmul(embedding, embedding.t())
    
#     # Step 3: Create the mask for positive and negative pairs
#     N = cosine_sim.size(0)
#     is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(1, N)).float()
#     is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(1, N)).float()
    
#     # Mask scores related to itself (diagonal elements)
#     is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    
#     # Step 4: Convert cosine similarity to angles
#     angles = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
    
#     # Step 5: Add margin to angles for positive pairs
#     angles_with_margin = angles + margin * is_pos
    
#     # Step 6: Convert angles back to cosine similarity
#     cosine_with_margin = torch.cos(angles_with_margin)
    
#     # Step 7: Compute positive and negative logits
#     s_p = cosine_with_margin * is_pos
#     s_n = cosine_sim * is_neg
    
#     # Step 8: Apply scaling factor and large negative values for masking
#     logit_p = gamma * s_p + (-99999999.) * (1 - is_pos)
#     logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)
    
#     # Step 9: Compute the loss using softplus and logsumexp
#     loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    
#     return loss
# def arcface_loss(
#         embedding: torch.Tensor,
#         targets: torch.Tensor,
#         margin: float,
#         gamma: float) -> torch.Tensor:
    
#     # Step 1: Normalize embedding features
#     embedding = F.normalize(embedding, dim=1)
    
#     # Step 2: Compute cosine similarity
#     cosine_sim = torch.matmul(embedding, embedding.t())
    
#     # Step 3: Convert cosine similarity to angles
#     angles = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
    
#     # Step 4: Add margin to angles
#     angles_with_margin = angles + margin
    
#     # Step 5: Convert angles back to cosine similarity
#     cosine_with_margin = torch.cos(angles_with_margin)
    
#     N = cosine_sim.size(0)
#     is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
#     is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    
#     # Mask scores related to itself
#     is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    
#     s_p = cosine_with_margin * is_pos
#     s_n = cosine_sim * is_neg
    
#     logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
#     logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)
    
#     loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    
#     return loss


# def arcface_loss(
#         embedding: torch.Tensor,
#         targets: torch.Tensor,
#         margin: float,
#         gamma: float) -> torch.Tensor:
#     # Normalize embedding features
#     embedding = F.normalize(embedding, dim=1)

#     # Compute cosine similarity matrix
#     cosine_sim = torch.matmul(embedding, embedding.t())

#     N = cosine_sim.size(0)
#     is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
#     is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

#     # Mask scores related to itself
#     is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

#     s_p = cosine_sim * is_pos
#     s_n = cosine_sim * is_neg

#     # Apply arc margin
#     theta_p = torch.acos(s_p.clamp(-1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical issues
#     theta_p += margin
#     s_p = torch.cos(theta_p)

#     logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
#     logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)

#     # Correctly compute the final loss
#     loss = -F.log_softmax(torch.cat((logit_p, logit_n), dim=1), dim=1)[:, :N].sum(dim=1).mean()

#     return loss

# def arcface_loss(
#         embedding: torch.Tensor,
#         targets: torch.Tensor,
#         margin: float,
#         gamma: float, ) -> torch.Tensor:
#     # Normalize embedding features
#     embedding = F.normalize(embedding, dim=1)

#     dist_mat = torch.matmul(embedding, embedding.t())

#     N = dist_mat.size(0)
#     is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
#     is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

#     # Mask scores related to itself
#     is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

#     # ArcFace modification
#     # dist_mat.acos_()  # Apply acos to convert cosine similarity to angle
#     s_p = dist_mat * is_pos
#     s_n = dist_mat * is_neg

#     # Add margin to positive pairs
#     s_p = s_p + margin

#     # Convert angles back to cosine values
#     s_p.cos_()
#     s_n.cos_()

#     # Apply scaling
#     logit_p = gamma * s_p + (-99999999.) * (1 - is_pos)
#     logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)
#     # print(logit_p,logit_n)

#     # Compute loss
#     loss = -F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

#     return loss


# def arcface_loss(
#         embedding: torch.Tensor,
#         targets: torch.Tensor,
#         margin: float,
#         gamma: float) -> torch.Tensor:
    
#     # Step 1: Normalize embedding features
#     embedding = F.normalize(embedding, dim=1)
    
#     # Step 2: Compute cosine similarity
#     cosine_sim = torch.matmul(embedding, embedding.t())
    
#     # Step 3: Convert cosine similarity to angles
#     angles = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
    
#     # Step 4: Add margin to angles
#     angles_with_margin = angles + margin
    
#     # Step 5: Convert angles back to cosine similarity
#     cosine_with_margin = torch.cos(angles_with_margin)
    
#     N = cosine_sim.size(0)
#     is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
#     is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    
#     # Mask scores related to itself
#     is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    
#     s_p = cosine_with_margin * is_pos
#     s_n = cosine_sim * is_neg
    
#     logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
#     logit_n = gamma * s_n + (-99999999.) * (1 - is_neg)
    
#     loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    
#     return loss
# def arcface_loss(cosine: torch.Tensor, label, s=64.0, m=0.5):
#     index = torch.where(label != -1)[0]
#     # print(index)
#     m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#     m_hot.scatter_(1, label[index, None], m)
#     cosine.acos_()
#     cosine[index] += m_hot
#     cosine.cos_().mul_(s)
#     return cosine

# 示例输入
# embedding = torch.randn(128, 512)  # 128 个样本，每个嵌入向量的维度为 512
# targets = torch.randint(0, 10, (128,))  # 128 个样本的目标标签，共 10 个类别
# print(embedding.shape, targets.shape)
# # 调用 arcface 函数
# margin = 0.5
# gamma = 64.0
# loss = arcface_loss(embedding, targets, margin, gamma)

# # # 打印输出损失
# import numpy as np
# print("Loss:", loss)





# import torch
# import torch.nn.functional as F

# def arcface_loss(embedding: torch.Tensor, targets: torch.Tensor, s=64.0, m=0.5) -> torch.Tensor:
#     # 正则化嵌入向量
#     embedding = F.normalize(embedding, dim=1)

#     # 计算嵌入向量之间的余弦相似度
#     cosine = torch.atmul(embedding, embedding.t())

#     # 截断余弦相似度，确保其值在合理范围内
#     cosine = torch.clamp(cosine, min=-1.0, max=1.0)

#     N = cosine.size(0)

#     # 创建目标标签独热编码
#     targets_one_hot = F.one_hot(targets, num_classes=N)

#     # ArcFace角度变换
#     cosine.acos_()
#     cosine += m * targets_one_hot.to(embedding.dtype)
#     cosine.cos_().mul_(s)

#     # ArcFace Loss计算
#     logit = cosine

#     loss = F.cross_entropy(logit, targets)

#     return loss

# # 示例调用
# embedding = torch.randn([128])  # 示例嵌入向量
# targets = torch.randint(0, 10, (128,2048))  # 示例目标标签，确保目标标签的范围正确

# print(embedding.shape)
# print(targets.shape)
# loss = arcface_loss(embedding, targets)
# print(loss)m


