import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import torchvision.models as models

class Attn_Net_Gated(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class GMA(nn.Module):
    def __init__(self, ndim=1024, gate = True, size_arg = "big", dropout = False, n_classes = 2, n_tasks=1):
        super(GMA, self).__init__()
        self.size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        
        initialize_weights(self)
    
    def get_sign(self, h):
        A, h = self.attention_net(h)# h: Bx512
        w = self.classifier.weight.detach()
        sign = torch.mm(h, w.t())
        return sign
    
    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A[0]
        
        A_raw = A.detach().cpu().numpy()[0]
        w = self.classifier.weight.detach()
        sign = torch.mm(h.detach(), w.t()).cpu().numpy()
        
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        
        logits  = self.classifier(M) 
        return A_raw, sign, logits

class TileCNNEncoder(nn.Module):
    def __init__(self, arch):
        super(TileCNNEncoder, self).__init__()
        model = models.__dict__[arch](weights="DEFAULT")
        if arch.startswith('efficient'):
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.ndim = model.classifier[1].in_features
        elif arch.startswith('resnet'):
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.ndim = model.fc.in_features
        elif arch.startswith('mobilenet_v2'):
            self.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
            self.ndim = model.classifier[1].in_features
        elif arch.startswith('mobilenet_v3'):
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.ndim = model.classifier[0].in_features
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), self.ndim)

class TileVITEncoder(nn.Module):
    def __init__(self, arch):
        super(TileVITEncoder, self).__init__()
        model = models.__dict__[arch](weights="DEFAULT", image_size=256)
        if arch.startswith('vit'):
            self.model = model
            self.ndim = model.heads[0].in_features
    
    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 0]
        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
