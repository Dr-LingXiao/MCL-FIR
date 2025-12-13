import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from layers import MultiheadLinear
import math
import copy
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class CCLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super(CCLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.temperature = temperature

        self.bs = None
        self.batch_indexes = None
        self.positive_mask = None
        self.negative_mask = None
        self.labels = None

        self.cluster_labels = None
        self.cluster_negative_mask = None
        self.cluster_positive_mask = None
        self.cluster_indexes = None

    def instance_loss(self, z1, z2):
        bs = z1.shape[0]

        if self.batch_indexes is None or self.bs != bs:
            self.bs = bs
            self.batch_indexes = torch.arange(bs, device=z1.device)
            self.positive_mask = F.one_hot(torch.cat([self.batch_indexes + bs, self.batch_indexes]), bs * 2).bool()
            self.negative_mask = (1 - F.one_hot(torch.cat([self.batch_indexes, self.batch_indexes + bs]), bs * 2)-self.positive_mask.float()).bool()
            self.labels = torch.zeros((2*bs,),device=z1.device).long()

        z = F.normalize(torch.cat([z1, z2], dim=0),p=2,dim=-1)
        s = z @ z.T

        positives = s[self.positive_mask].view(2 * bs, 1)
        negatives = s[self.negative_mask].view(2 * bs, -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        loss = self.CE(logits, self.labels)
        return loss

    def cluster_loss(self, p1, p2):
        if len(p1.shape)==2:
            p1 = p1.unsqueeze(0)
        if len(p2.shape)==2:
            p2 = p2.unsqueeze(0)
        k,bs,c = p1.shape
        if self.cluster_indexes is None or self.bs != bs:
            self.bs = bs
            self.cluster_indexes = torch.arange(c, device=p1.device)
            self.cluster_positive_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes + c, self.cluster_indexes]), c * 2).bool()])
            negative_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes, self.cluster_indexes + c]), c * 2)])
            self.cluster_negative_mask = (1 - negative_mask - self.cluster_positive_mask.float()).bool()
            self.cluster_labels = torch.zeros((2*c,),device=p1.device).long()

        p = torch.cat([p1, p2], dim=2)
        if len(p.shape)==2:
            p = p.unsqueeze(0)
        p = F.normalize(p,dim=1)
        s = torch.einsum("kna, knb->kab", p, p)
        positives = s[self.cluster_positive_mask].view(k,2*c,-1)
        negatives = s[self.cluster_negative_mask].view(k,2*c,-1)
        logits = torch.cat([positives, negatives], dim=2)

        loss_ce = []
        loss_ne = []
        for k_ in range(k):
            loss_ce.append(self.CE(logits[k_], self.cluster_labels))
            p_i = p1[k_].sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = p2[k_].sum(0).view(-1)
            p_j /= p_j.sum()
            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            loss_ne.append(ne_i + ne_j)
        return loss_ce, loss_ne

    def forward(self, p1, p2, z1,z2):
        # loss_ce, loss_ne = self.cluster_loss(p1,p2)
        loss_cc = self.instance_loss(z1, z2)
        return loss_cc

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, embedding_size, cls_num):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.embedding_size = embedding_size
        self.cls_num = cls_num
        self.criterion = nn.BCEWithLogitsLoss()
        self.CCLoss = CCLoss()
        self.id_mlp = nn.Sequential(nn.Linear(self.embedding_size, 64))
        self.clu_mlp = nn.Sequential(MultiheadLinear(self.embedding_size, self.cls_num, 1, True))
        self.kd_loss = nn.MSELoss()
        
    def update_teacher(self):
        """ 在训练时调用，确保 EMA 教师模型的参数更新 """
        self.teacher.update()
    def forward(self, x, y, z, c, c0, c1, x1, y1, z1):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which attribute images are compared"""
        embedded_x, student_x, teacher_x = self.embeddingnet(x, c0, c1, x1)  # pure, noise
        embedded_y, student_y, teacher_y = self.embeddingnet(y, c0, c1, y1)
        embedded_z, student_z, teacher_z = self.embeddingnet(z, c0, c1, z1)
        sim_a = torch.sum(embedded_x * embedded_y, dim=1)
        sim_b = torch.sum(embedded_x * embedded_z, dim=1)
        # with torch.no_grad():
       
            
        loss_kd = self.kd_loss(student_x, teacher_x.detach()) + self.kd_loss(student_y, teacher_y.detach()) + self.kd_loss(student_z, teacher_z.detach())
        
      

        p1, p2 = F.softmax(self.clu_mlp(embedded_x), dim=-1), F.softmax(self.clu_mlp(embedded_z), dim=-1)
        # z1, z2 = self.id_mlp(embedded_x), self.id_mlp(embedded_z)
        loss_cc = self.CCLoss(p1, p2, embedded_x, embedded_z)

     


        loss_c = loss_cc 
        # c_1 = torch.eye(2)[0].repeat(c.size(0), 1)
        # c_2 = torch.eye(2)[1].repeat(c.size(0), 1)
        #
        # c_1 = c_1.cuda()
        # c_2 = c_2.cuda()

        # loss_noise = (self.criterion(x1, c_1) + self.criterion(y1, c_1) + self.criterion(z1, c_1) + self.criterion(x_d1,
        #                                                                                                            c_2) + self.criterion(
        #     y_d1, c_2) + self.criterion(z_d1, c_2) +
        #               self.criterion(x1_r, c_1) + self.criterion(y1_r, c_1) + self.criterion(z1_r,
        #                                                                                      c_1) + self.criterion(
        #             x_d1_r, c_2) + self.criterion(y_d1_r, c_2) + self.criterion(z_d1_r, c_2)) / 12.

        return sim_a, sim_b, loss_c, loss_kd


class EMA(nn.Module):
    """ Exponential Moving Average (EMA) for model weights """
    def __init__(self, backbonenet, decay, device='cuda'):
        super(EMA, self).__init__()
        self.decay = decay
        self.device = device
        self.model = backbonenet  # ✅ 存储原始模型
        
        self.ema_model = self._copy_model(backbonenet)
        self.ema_model.to(self.device)  # ✅ 迁移 `ema_model` 到 GPU
        self.ema_model.eval()  # ✅ 让 `ema_model` 处于推理模式，避免 BatchNorm/Dropout 影响

    

    def _copy_model(self, backbonenet):
        
        """Deep copy model to maintain EMA"""
        ema_model = copy.deepcopy(backbonenet)  # ✅ 直接深拷贝，避免手动初始化
        ema_model.load_state_dict(backbonenet.state_dict())  # ✅ 复制 `model` 参数到 `ema_model`
        for param in ema_model.parameters():
            param.requires_grad = False  # ✅ 冻结 EMA 模型参数
        return ema_model


    def update(self):
        """Update EMA model weights"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data

    def forward(self, x):
        """ ✅ 让 `self.teacher(x, c0, c1)` 调用 `ema_model.forward()` """
        return self.ema_model(x)  # ✅ 直接让 `ema_model` 计算前向传播


# class test(nn.Module):
#     def __init__(self, embeddingnet, decay, device='cuda'):
#         super(test, self).__init__()
#         self.device = device
#         self.embeddingnet = embeddingnet.to(self.device)
#         self.teacher = EMA(
#             self.embeddingnet, 
#             embeddingnet.backbonenet, 
#             embeddingnet.embedding_size, 
#             embeddingnet.n_attributes, 
#             decay=decay, 
#             device=device
#         )  # ✅ `EMA` 现在可以正常使用

#     def forward(self, x):
#         """ x: Anchor image """
#         embedded_x = self.embeddingnet(x.to(self.device))  
#         embedded_x_1 = self.teacher(x)  # ✅ 现在不会报错
#         return (embedded_x + embedded_x_1) / 2.
            
class ASENet(nn.Module):
    def __init__(self, backbonenet, embedding_size, n_attributes):
        super(ASENet, self).__init__()
        self.backbonenet = backbonenet
        self.n_attributes = n_attributes
        self.embedding_size = embedding_size

        self.mask_fc1 = nn.Linear(self.n_attributes, 512, bias=False)
        self.mask_fc2 = nn.Linear(self.n_attributes, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.feature_fc = nn.Linear(1024, 1024)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, c, norm=True):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)

        x = x * attmap
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = x.sum(dim=2)

        mask = self.ACA(x, c)

        x = x * mask

        x = self.feature_fc(x)

        if norm:
            x = l2norm(x)

        return x

    def ASA(self, x, c):
        # attribute-aware spatial attention
        img_embedding = self.conv1(x)
        img_embedding = self.tanh(img_embedding)

        c = c.view(c.size(0), 1).cpu()
        mask_fc_input = torch.zeros(c.size(0), self.n_attributes).scatter_(1, c, 1)
        mask_fc_input = mask_fc_input.cuda()
        mask = self.mask_fc1(mask_fc_input)
        mask = self.tanh(mask)
        mask = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask = mask.expand(mask.size(0), mask.size(1), 14, 14)

        attmap = mask * img_embedding
        attmap = self.conv2(attmap)
        attmap = self.tanh(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)

        return attmap

    def ACA(self, x, c):
        # attribute-aware channel attention
        c = c.view(c.size(0), 1).cpu()
        mask_fc_input = torch.zeros(c.size(0), self.n_attributes).scatter_(1, c, 1)
        mask_fc_input = mask_fc_input.cuda()
        mask = self.relu(self.mask_fc2(mask_fc_input))
        mask = torch.cat((x, mask), dim=1)
        mask = self.fc1(mask)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)

        return mask

    def get_heatmaps(self, x, c):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)
        attmap = attmap.squeeze()

        return attmap


class ASENet_V2(nn.Module):
    def __init__(self, backbonenet, embedding_size, n_attributes, decay):
        super(ASENet_V2, self).__init__()
        self.backbonenet = backbonenet
        self.n_attributes = n_attributes
        self.embedding_size = embedding_size
        self.teacher = EMA(self.backbonenet, decay).to('cuda')  # EMA teacher model
        # self.attr_embedding = torch.nn.ModuleList()
        # self.attr_embedding.append(torch.nn.Embedding(1, 512))
        # for t in range(1, self.n_attributes):
        #     self.attr_embedding.append(torch.nn.Embedding(2, 512))

        # if self.n_attributes == 1:
        #     self.attr_embedding1 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 2:
        #     self.attr_embedding2 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 3:
        #     self.attr_embedding3 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 4:
        #     self.attr_embedding4 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 5:
        #     self.attr_embedding5 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 6:
        #     self.attr_embedding6 = torch.nn.Embedding(n_attributes, 512)
        # elif self.n_attributes == 7:
        #     self.attr_embedding7 = torch.nn.Embedding(n_attributes, 512)

        self.attr_transform1 = nn.Linear(512, 128)
        self.conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1)
        self.img_bn1 = nn.BatchNorm2d(128)

        self.attr_transform2 = nn.Linear(512, 128)
        self.fc1 = nn.Linear(256, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.desired_mean = 0.0
        self.desired_stddev = 0.1

        # self.fc_class= nn.Linear(512, 2)
        #
        # self.gaussian_noise_tensor = nn.Parameter(torch.randn(1, 512) * self.desired_stddev + self.desired_mean)

    def forward(self, x, c0, c1, test=False, x1=None):
        if x1 is not None: 
            x_1 = self.teacher(x1)
        else:
            x_1 = self.teacher(x)
        x_raw = self.backbonenet(x)
        
        x = self.conv1(x_raw)

        attmap = self.ASA(x, c0, c1)
        x = x * attmap

        ################################
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = x.sum(dim=2)

        mask = self.ACA(x, c0, c1)
        x = x * mask

       
        x = l2norm(x)
        if test is True:
            return x
        
        x_raw = x_raw.view(x_raw.size(0), x_raw.size(1), -1).mean(dim=-1)  # 对 W*H 维度求均值
        x_1 = x_1.view(x_1.size(0), x_1.size(1), -1).mean(dim=-1)  # 对 W*H 维度求均值
        
        return x, x_raw, x_1

    def ASA(self, x, attr0, attr1):
        # attribute-aware spatial attention
        img = self.img_bn1(x)
        img = self.tanh(img)
        attr = torch.add(attr0, attr1)
        attr = self.attr_transform1(attr)

        attr = attr.view(attr.size(0), attr.size(1), 1, 1)
        attr = attr.expand(attr.size(0), attr.size(1), 14, 14)
        attmap = attr * img
        attmap = torch.sum(attmap, dim=1, keepdim=True)
        attmap = torch.div(attmap, 128 ** 0.5)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)
        ################################

        return attmap

    def ACA(self, x, attr0, attr1):

        attr = torch.add(attr0, attr1)
        attr = self.attr_transform1(attr)
        img_attr = torch.cat((x, attr), dim=1)
        mask = self.fc1(img_attr)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)

        return mask


    def get_heatmaps(self, x, attr0, attr1):
        x = self.backbonenet(x)

        attmap = self.ASA(x, attr0, attr1)

        attmap = attmap.squeeze()

        return attmap

    
class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, embedding_size, n_attributes, learnedmask=True, prein=False):
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.embed_fc = nn.Linear(1024, embedding_size)
        self.avgpool = nn.AvgPool2d(14)
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_attributes, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_attributes)
                for i in range(n_attributes):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_attributes, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_attributes, embedding_size])
            mask_len = int(embedding_size / n_attributes)
            for i in range(n_attributes):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c, norm=True):
        embedded_x = self.embeddingnet(x)
        embedded_x = self.avgpool(embedded_x)
        embedded_x = embedded_x.view(embedded_x.size(0), -1)
        embedded_x = self.embed_fc(embedded_x)
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask

        if norm:
            masked_embedding = l2norm(masked_embedding)
            
        return masked_embedding
    

model_dict = {
    'Tripletnet': Tripletnet,
    'ASENet': ASENet,
    'ASENet_V2': ASENet_V2,
    'ConditionalSimNet': ConditionalSimNet
}
def get_model(name):
    return model_dict[name]
