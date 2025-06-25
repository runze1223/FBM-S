__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


from layers.FBMS_backbone import backbone_PatchTST
from layers.FBMS_backbone import Base_seasonal
from layers.FBMS_backbone import Interaction_backbone
from layers.FBMS_backbone import MLP_backbone
from layers.FBMS_backbone import MLP_backbone_patch
import math
from torch.fft import rfft, irfft
from layers.VH_plus import VH
from layers.RevIN import RevIN

from torch.nn.functional import gumbel_softmax
from einops import rearrange
from torch.distributions.normal import Normal

class BinaryConcrete(nn.Module):
    def __init__(self, temp):
        super(BinaryConcrete, self).__init__()
        self.temp = temp
        self.sigmoid = nn.Sigmoid()

    def forward(self,alpha):
        noise = torch.rand_like(alpha).cuda()
        noise=torch.log(noise)-torch.log(1-noise)
        ouput=self.sigmoid((alpha + noise) / self.temp)

        return ouput


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)

        # B x C x C x D
        diff = X1 - X2

        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)

        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)
        # 对角线置零

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)

        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)

        # bernoulli中两个通道有关系的概率
        sample = self.bernoulli_gumbel_rsample(p)

        mask = sample.unsqueeze(1)
        cnt = torch.sum(mask, dim=-1)
        return mask


def top_k_amplitudes_torch(x, k):
    amplitudes = torch.abs(x)        # Take absolute value (amplitude)
    values, indices = torch.topk(amplitudes, k, dim=-1)  # Get top-k amplitudes and their indices
    mask = torch.zeros_like(x, dtype=amplitudes.dtype)  # create mask with same shape as x, filled with 0
    # scatter 1.0 to the top-k positions
    mask.scatter_(dim=-1, index=indices, value=1.0)

    mask2 = torch.ones_like(x, dtype=amplitudes.dtype)  # create mask with same shape as x, filled with 0
    # scatter 1.0 to the top-k positions
    mask2.scatter_(dim=-1, index=indices, value=0.0)



    return mask, mask2
#############################################################################################

###########################################################################################

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        super().__init__()   
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_model2 = configs.d_model2
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        dropout2=configs.dropout2
        
        individual = configs.individual
        drop_initial=configs.drop_initial
        centralization=configs.centralization
        

        patch_len = configs.patch_len
        patch_num = configs.patch_num
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine

        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        multiscale=configs.multiscale

        sr=context_window
        self.context_window=context_window
        self.revin=revin
        self.channel_mask=configs.channel_mask

        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        ts = 1.0/sr
        t = np.arange(0,1,ts)
        t=torch.tensor(t).cuda()
        for i in range(self.context_window//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                if i==(self.context_window//2+1):
                    cos=torch.vstack([cos,0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)])
                else:
                    cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)])

        self.cos = nn.Parameter(cos, requires_grad=False)
        self.sin = nn.Parameter(sin, requires_grad=False)

        self.timestamp=configs.timestamp
        self.seasonal=configs.seasonal
        self.interaction=configs.interaction
        self.trend=configs.trend
        self.patch=configs.patch

        if self.timestamp==1:
            self.VH=VH(configs)


        cut1=configs.cut1
        cut2=configs.cut2
        hidden1=configs.hidden1
        hidden2=configs.hidden2
        linear=configs.linear
        
        self.self_backbone=configs.self_backbone

        self.dropout=nn.Dropout(p=configs.dropout_total)

        self.dropout2=nn.Dropout(p=configs.dropout_total2)

        drop_initial=configs.drop_initial

        if self.channel_mask:
            self.mask_generator = Mahalanobis_mask(configs.seq_len)

        

        # W_pos = torch.empty(context_window//2+1)
        # nn.init.uniform_(W_pos, -0.001, 0.001)
        # self.parameter=nn.Parameter(W_pos, requires_grad=True)
        # W_pos2 = torch.empty(context_window//2+1)
        # nn.init.uniform_(W_pos2, -0.001, 0.001)
        # self.parameter2=nn.Parameter(W_pos2, requires_grad=True)

        if self.interaction==1:
            self.model_interaction=Interaction_backbone(configs, context_window, target_window,cut1 ,cut2,d_model2,dropout2,n_heads,n_layers)

        # self.temp=0.05
        # self.binaryConcrete=BinaryConcrete(self.temp)
        if self.seasonal==1:
            self.model_seasonal=Base_seasonal(context_window,target_window,multiscale)

        if self.self_backbone=="PatchTST":
            self.model = backbone_PatchTST(c_in=c_in, context_window = context_window, target_window=target_window, patch_num=patch_num, stride=stride, 
                            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                            n_heads=n_heads,multiscale=multiscale,linear=linear,  d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                            pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                            subtract_last=subtract_last, verbose=verbose,drop_initial=drop_initial,centralization=centralization, **kwargs)
        else:
            if self.patch==1:
                self.model= MLP_backbone_patch(c_in,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale, drop_initial,patch_num,centralization)
            else:
                self.model= MLP_backbone(context_window, target_window,dropout,hidden1,hidden2,linear,multiscale, drop_initial)

    # def sample(self, alpha, temp=None):
    #     if self.training:
    #         residual=self.binaryConcrete(alpha)
    #         return residual
    #     else:
    #         return(torch.sigmoid((alpha)/self.temp)> 0.5).float()



    def forward(self, x,y,z):           # x: [Batch, Input length, Channel]


        if self.timestamp==1:
            x,adding,kl_divergence_total=self.VH(x,y,z)
        
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

        if self.channel_mask:
            channel_mask = self.mask_generator(x)
        else:
            channel_mask=None
            
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)

        norm=x.size()[-1]
        frequency=rfft(x,axis=-1)
        X_oneside=frequency/(norm)*2

        # mask1, mask2 =top_k_amplitudes_torch(X_oneside,20)

        basis_cos=torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos)
        basis_sin=torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin)
        x=basis_cos+basis_sin

        # trend=torch.einsum('bkpt,bkp->bkpt', x, mask1)
        # interaction=torch.einsum('bkpt,bkp->bkpt', x, mask2)

        adds=[]
        if self.trend==1:
            add = self.dropout(self.model(x))
            adds.append(add)

        if self.seasonal==1:
            add=self.model_seasonal(x,X_oneside)
            adds.append(add)

        if self.interaction==1:
            add=self.model_interaction(x,channel_mask)
            add= self.dropout2(add)
            adds.append(add)

        for i in range(len(adds)):
            if i==0:
                k=adds[i]
            else:
                k=k+adds[i]

        if self.revin: 
            k = k.permute(0,2,1)
            k = self.revin_layer(k, 'denorm')
            k = k.permute(0,2,1)

        k = k.permute(0,2,1)    # x: [Batch, Input length, Channel]

        if self.timestamp==1:
            for i in range(len(adding)):
                k=adding[i].permute(0,2,1)+k

        return k


    # def obtain(self, x,y,z):           # x: [Batch, Input length, Channel]

    #     if self.timestamp==1:
    #         x,adding,kl_divergence_total=self.VH(x,y,z)
        
    #     x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

    #     if self.channel_mask:
    #         channel_mask = self.mask_generator(x)
    #     else:
    #         channel_mask=None
            
    #     if self.revin: 
    #         x = x.permute(0,2,1)
    #         x = self.revin_layer(x, 'norm')
    #         x = x.permute(0,2,1)

    #     norm=x.size()[-1]
    #     frequency=rfft(x,axis=-1)
    #     X_oneside=frequency/(norm)*2

    #     # mask1, mask2 =top_k_amplitudes_torch(X_oneside,20)

    #     basis_cos=torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos)
    #     basis_sin=torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin)
    #     x=basis_cos+basis_sin

    #     # trend=torch.einsum('bkpt,bkp->bkpt', x, mask1)
    #     # interaction=torch.einsum('bkpt,bkp->bkpt', x, mask2)

    #     adds=[]
    #     if self.trend==1:
    #         add = self.dropout(self.model(x))
    #         adds.append(add)

    #     if self.seasonal==1:
    #         add=self.model_seasonal(x,X_oneside)
    #         adds.append(add)

    #     if self.interaction==1:
    #         add=self.model_interaction(x,channel_mask)
    #         add= self.dropout2(add)
    #         adds.append(add)

    #     for i in range(len(adds)):
    #         if i==0:
    #             k=adds[i]
    #         else:
    #             k=k+adds[i]

    #     if self.revin: 
    #         k = k.permute(0,2,1)
    #         k = self.revin_layer(k, 'denorm')
    #         k = k.permute(0,2,1)

    #     k = k.permute(0,2,1)    # x: [Batch, Input length, Channel]

    #     if self.timestamp==1:
    #         for i in range(len(adding)):
    #             k=adding[i].permute(0,2,1)+k

    #     mean=self.revin_layer.mean
    #     std=self.revin_layer.stdev


    #     return adds, mean, std
