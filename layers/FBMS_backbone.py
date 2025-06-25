__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from math import sqrt

from layers.PatchTST_layers import *
from layers.RevIN import RevIN

import math
from torch.fft import rfft, irfft

from layers.expert_moe import Linear_extractor_cluster

from einops import rearrange

class Base_seasonal(nn.Module):
    def __init__(self,context_window, target_window,multiscale):
        super().__init__()
        self.context_window=context_window
        sr=self.context_window
        self.multiscale=multiscale
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

        cos=cos[1:]
        sin=sin[1:]

        rolled_tensor_cos = torch.stack([cos.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)
        rolled_tensor_sin = torch.stack([sin.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)

        self.cos = nn.Parameter(rolled_tensor_cos, requires_grad=False)
        self.sin = nn.Parameter(rolled_tensor_sin, requires_grad=False)

        # rolled_tensor_cos2 = torch.stack([cos.roll(shifts=-2*i,dims=-1) for i in range(target_window//2)], dim=0)
        # rolled_tensor_sin2 = torch.stack([sin.roll(shifts=-2*i,dims=-1) for i in range(target_window//2)], dim=0)

        # self.cos2 = nn.Parameter(rolled_tensor_cos2, requires_grad=False)
        # self.sin2 = nn.Parameter(rolled_tensor_sin2, requires_grad=False)

        # rolled_tensor_cos3 = torch.stack([cos.roll(shifts=-4*i,dims=-1) for i in range(target_window//4)], dim=0)
        # rolled_tensor_sin3 = torch.stack([sin.roll(shifts=-4*i,dims=-1) for i in range(target_window//4)], dim=0)

        # self.cos3 = nn.Parameter(rolled_tensor_cos3, requires_grad=False)
        # self.sin3 = nn.Parameter(rolled_tensor_sin3, requires_grad=False)


        W_pos = torch.empty((self.context_window//2,self.context_window))
        nn.init.uniform_(W_pos, -0.001, 0.001)
        self.parameter=nn.Parameter(W_pos, requires_grad=True)
        # W_pos2 = torch.empty((self.context_window//2,self.context_window))
        # nn.init.uniform_(W_pos2, -0.001, 0.001)
        # self.parameter2=nn.Parameter(W_pos2, requires_grad=True)

        # W_pos2 = torch.empty((self.context_window//4,self.context_window//2))
        # nn.init.uniform_(W_pos2, -0.001, 0.001)
        # self.parameter2=nn.Parameter(W_pos2, requires_grad=True)

        # W_pos3 = torch.empty((self.context_window//8,self.context_window//4))
        # nn.init.uniform_(W_pos3, -0.001, 0.001)
        # self.parameter3=nn.Parameter(W_pos3, requires_grad=True)
        

    def forward(self, x,freq):   
                                      # x: [bs x nvars x d_model x patch_num]

        x=x[:,:,1:,:]
        freq=freq[:,:,1:]


        hidden_cos=torch.einsum('pkt,kt->pk', self.cos,  self.parameter)
        hidden_sin=torch.einsum('pkt,kt->pk', self.sin,  self.parameter)
        x=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin)


        # if self.multiscale==2:

        #     parameter2 =self.parameter2.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        #     hidden_cos2=torch.einsum('pkt,kt->pk', self.cos2,  0.5*parameter2)
        #     hidden_sin2=torch.einsum('pkt,kt->pk', self.sin2,  0.5*parameter2)
        #     add1=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos2)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin2)

        #     parameter3 =self.parameter3.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)

        #     hidden_cos3=torch.einsum('pkt,kt->pk', self.cos3,  0.25*parameter3)
        #     hidden_sin3=torch.einsum('pkt,kt->pk', self.sin3,  0.25*parameter3)
        #     add2=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos3)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin3)


        #     add1 = F.interpolate(add1, scale_factor=2, mode='linear', align_corners=True)  
        #     add2 = F.interpolate(add2, scale_factor=4, mode='linear', align_corners=True)

        #     x=x+add1+add2
        # elif self.multiscale==1:
        #     parameter2 =self.parameter2.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        #     hidden_cos2=torch.einsum('pkt,kt->pk', self.cos2,  0.5*parameter2)
        #     hidden_sin2=torch.einsum('pkt,kt->pk', self.sin2,  0.5*parameter2)
        #     add1=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos2)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin2)
        #     add1 = F.interpolate(add1, scale_factor=2, mode='linear', align_corners=True)
        #     x=x+add1
    
        return x


class MLP_backbone(nn.Module):
    def __init__(self,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale,drop_initial):
        super().__init__()

        self.context_window=context_window
        self.target_window=target_window

        self.drop_initial=drop_initial

        self.linear=linear
        self.multiscale=multiscale

        self.flatten = nn.Flatten(start_dim=-2)

        if self.linear==1:
            if self.multiscale==2:
                self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
                self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
                self.linear3 = nn.Linear((self.context_window//4)*(self.context_window//8),target_window)
            elif self.multiscale==1:
                self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
                self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
            else:
                self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
        else:
            if self.multiscale==2:
                self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )

                self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
                self.linear3 = nn.Sequential(   nn.Linear((self.context_window//4)*(self.context_window//8),hidden1), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
            elif self.multiscale==1:
                self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )

                self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
            else:
                self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )

    def forward(self, x):                             
        x=x[:,:,1:,:]
        if self.multiscale==1:
            down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1)

            if self.drop_initial:
                add2=self.linear2(self.flatten(down1))
            else:
                add1=self.linear1(self.flatten(x))
                add2=self.linear2(self.flatten(down1))

            if self.drop_initial:
                 x=add2
            else:
                 x=add1+add2

        elif self.multiscale==2:
            down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1) 

            down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
            down2= down2.sum(dim=-2) 
            down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
            down2= down2.mean(dim=-1)

            if self.drop_initial:
                add2=self.linear2(self.flatten(down1))     
                add3=self.linear3(self.flatten(down2))
            else:
                add1=self.linear1(self.flatten(x))
                add2=self.linear2(self.flatten(down1))     
                add3=self.linear3(self.flatten(down2))
            if self.drop_initial:
                 x=add2+add3
            else:
                 x=add1+add2+add3
        else:
            x=self.linear1(self.flatten(x))
        return x

class MLP_backbone_patch(nn.Module):
    def __init__(self,c_in,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale,drop_initial,pacth_num,centralization):
        super().__init__()

        self.revin1 = RevIN(c_in)
        self.revin2 = RevIN(c_in)
        self.revin3 = RevIN(c_in)
        self.centralization=centralization

        self.context_window=context_window
        self.target_window=target_window
        self.pacth_num=pacth_num

        self.drop_initial=drop_initial

        self.linear=linear
        self.multiscale=multiscale

        self.flatten = nn.Flatten(start_dim=-2)

        patch_len=int((context_window*(context_window//2))/self.pacth_num)

        # self.noc_emb = nn.Parameter(torch.zeros(c_in, 128))

        if self.linear==1:
            if self.multiscale==2:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                self.linear2 = nn.Linear(patch_len//4,hidden1) 
                self.linear3 = nn.Linear(patch_len//16,hidden1) 
                self.linear11 = nn.Linear(hidden1*self.pacth_num,target_window)
                self.linear22 = nn.Linear(hidden1*self.pacth_num,target_window)
                self.linear33 = nn.Linear(hidden1*self.pacth_num,target_window)
            elif self.multiscale==1:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                self.linear2 = nn.Linear(patch_len//4,hidden1) 
                self.linear11 = nn.Linear(hidden1*self.pacth_num,target_window)
                self.linear22 = nn.Linear(hidden1*self.pacth_num,target_window)
            else:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                self.linear11 = nn.Linear(hidden1*self.pacth_num,target_window)
        else:
            if self.multiscale==2:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                self.linear11 = nn.Sequential( 
                    #  nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )

                self.linear2 = nn.Linear(patch_len//4,hidden1) 

                self.linear22 =nn.Sequential(   
                    # nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
                
                self.linear3 = nn.Linear(patch_len//16,hidden1) 
                
                self.linear33 = nn.Sequential(  
                    # nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
            elif self.multiscale==1:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                        
                self.linear11 = nn.Sequential(  
                    # nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )

                self.linear2 = nn.Linear(patch_len//4,hidden1) 

                self.linear22 =nn.Sequential(   
                                                # nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
            else:
                self.linear1 = nn.Linear(patch_len,hidden1) 
                        
                self.linear11 = nn.Sequential( 
                                                # nn.Linear(hidden1*self.pacth_num,hidden1*self.pacth_num), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden1*self.pacth_num,hidden2), 
                                                nn.Dropout(p=dropout),
                                                nn.ReLU(),
                                                nn.Linear(hidden2, target_window),
                                            )
                

                                            

    def forward(self, x):                             
        x=x[:,:,1:,:]
        if self.multiscale==1:
            down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1)

            a,b,c,d = down1.shape
            e=d//self.pacth_num
            down1=down1.reshape(a,b,c,self.pacth_num,e)
            down1=down1.permute(0,1,3,2,4)
            down1=down1.reshape(a,b,down1.size()[2],-1)

            if self.centralization:   
                down1 = rearrange(down1, "x y l c -> (x l) c y")
                down1 = self.revin1(down1, 'norm')
                down1 = rearrange(down1,  "(x l) c y -> x y l c", l=self.pacth_num)

            a,b,c,d = x.shape
            e=d//self.pacth_num
            x=x.reshape(a,b,c,self.pacth_num,e)
            x=x.permute(0,1,3,2,4)
            x=x.reshape(a,b,x.size()[2],-1) 
                   
            if self.centralization:   
                x = rearrange(x, "x y l c -> (x l) c y")
                x = self.revin3(x, 'norm')
                x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)

            if self.drop_initial:
                down1=self.linear2(down1)

                if self.centralization:     
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)

                down1=self.flatten(down1)
                # noc_emb = self.noc_emb.unsqueeze(0).repeat(a, 1, 1)
                # down1 = torch.cat([down1, noc_emb], dim=-1)

                add2=self.linear22(down1)
            else:

                x=self.linear1(x)
                if self.centralization:       
                    x = rearrange(x, "x y l c -> (x l) c y")
                    x = self.revin3(x, 'denorm')
                    x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)

                add1=self.linear11(self.flatten(x))
                down1=self.linear2(down1)


                if self.centralization:            
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                
                add2=self.linear22(self.flatten(down1))

            if self.drop_initial:
                 x=add2
            else:
                 x=add1+add2

        elif self.multiscale==2:
            down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1) 

            a,b,c,d = down1.shape
            e=d//self.pacth_num
            down1=down1.reshape(a,b,c,self.pacth_num,e)
            down1=down1.permute(0,1,3,2,4)
            down1=down1.reshape(a,b,down1.size()[2],-1)

            if self.centralization:    
                down1 = rearrange(down1, "x y l c -> (x l) c y")
                down1 = self.revin1(down1, 'norm')
                down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)

            down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
            down2= down2.sum(dim=-2) 
            down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
            down2= down2.mean(dim=-1)

            a,b,c,d = down2.shape
            e=d//self.pacth_num
            down2=down2.reshape(a,b,c,self.pacth_num,e)
            down2=down2.permute(0,1,3,2,4)
            down2=down2.reshape(a,b,down2.size()[2],-1)

            if self.centralization:     
                down2 = rearrange(down2, "x y l c -> (x l) c y")
                down2 = self.revin2(down2, 'norm')
                down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num)

            a,b,c,d = x.shape
            e=d//self.pacth_num
            x=x.reshape(a,b,c,self.pacth_num,e)
            x=x.permute(0,1,3,2,4)
            x=x.reshape(a,b,x.size()[2],-1)

            if self.centralization:             
                x = rearrange(x, "x y l c -> (x l) c y")
                x = self.revin3(x, 'norm')
                x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)

            if self.drop_initial:
                down1=self.linear2(down1)

                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)

                add2=self.linear22(self.flatten(down1))
                down2=self.linear3(down2)

                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'denorm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 

                add3=self.linear33(self.flatten(down2))
            else:
                x=self.linear1(x) 

                if self.centralization:     
                    x = rearrange(x, "x y l c -> (x l) c y")
                    x = self.revin3(x, 'denorm')
                    x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)  

                add1=self.linear11(self.flatten(x))
                down1=self.linear2(down1)

                if self.centralization:   
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)

                add2=self.linear22(self.flatten(down1))

                down2=self.linear3(down2)

                if self.centralization:    
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'denorm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 

                add3=self.linear33(self.flatten(down2))
            if self.drop_initial:
                 x=add2+add3
            else:
                 x=add1+add2+add3
        else:

           
            a,b,c,d = x.shape
            e=d//self.pacth_num
            x=x.reshape(a,b,c,self.pacth_num,e)
            x=x.permute(0,1,3,2,4)
            x=x.reshape(a,b,x.size()[2],-1)


            if self.centralization: 
                x = rearrange(x, "x y l c -> (x l) c y")
                x = self.revin3(x, 'norm')
                x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)


            x=self.linear1(x)
            
            if self.centralization: 
                x = rearrange(x, "x y l c -> (x l) c y")
                x = self.revin3(x, 'denorm')
                x = rearrange(x, "(x l) c y -> x y l c",l=self.pacth_num)

            x=self.flatten(x)            
            # noc_emb = self.noc_emb.unsqueeze(0).repeat(a, 1, 1)
            # x = torch.cat([x, noc_emb], dim=-1)

            x=self.linear11(x)
        return x





class backbone_PatchTST(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_num:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16,multiscale=0,linear=1, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False,drop_initial=False,centralization=True, **kwargs):
        
        super().__init__()

        self.context_window=context_window
        self.n_vars = c_in
        self.individual = individual

        self.pacth_num=patch_num
        self.target_window=target_window
        patch_len=int((context_window*(context_window//2))/self.pacth_num)

        self.linear=linear
        self.multiscale= multiscale

        self.drop_initial=drop_initial


        self.revin1 = RevIN(c_in)
        self.revin2 = RevIN(c_in)
        self.revin3 = RevIN(c_in)
        self.centralization=centralization


        # if self.linear==1:
        #     if self.multiscale==2:
        #         self.W_P_1 = nn.Linear(patch_len,d_model)
        #         self.W_P_2 = nn.Linear(patch_len//4,d_model)
        #         self.W_P_3 = nn.Linear(patch_len//16,d_model)
        #     elif self.multiscale==1:
        #         self.W_P_1  = nn.Linear(patch_len,d_model)
        #         self.W_P_2  = nn.Linear(patch_len//4,d_model)
        #     else:
        #         self.W_P_1 = nn.Linear(patch_len,d_model)


        # else:
        #     if self.multiscale==2:
        #         self.W_P_1 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )

        #         self.W_P_2 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len//4 ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )
        #         self.W_P_3 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len//16 ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )                                    
        #     elif self.multiscale==1:
        #         self.W_P_1 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )

        #         self.W_P_2 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len//4 ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )
        #     else:
        #         self.W_P_1 = nn.Sequential(
        #                                 torch.nn.Linear(patch_len ,d_model
        #                                 ),
        #                                 nn.Dropout(p=0.15),
        #                                 nn.ReLU(),
        #                                 # torch.nn.Linear( d_model,d_model)
        #                                 )

        if self.multiscale==2:
            self.W_P_1 = nn.Linear(patch_len,d_model)
            self.W_P_2 = nn.Linear(patch_len//4,d_model)
            self.W_P_3 = nn.Linear(patch_len//16,d_model)
        elif self.multiscale==1:
            self.W_P_1  = nn.Linear(patch_len,d_model)
            self.W_P_2  = nn.Linear(patch_len//4,d_model)
        else:
            self.W_P_1 = nn.Linear(patch_len,d_model)

                
        if self.multiscale==2:
            self.W_P_11 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )

            self.W_P_22 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )
            self.W_P_33 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )                                    
        elif self.multiscale==1:
            self.W_P_11 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )

            self.W_P_22 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )
        else:
            self.W_P_11 = nn.Sequential(
                                    nn.Dropout(p=0.15),
                                    nn.ReLU(),
                                    # torch.nn.Linear( d_model,d_model)
                                    )


        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.head_nf = d_model * self.pacth_num
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):                                                                 
        z=z[:,:,1:,:]

        if self.multiscale==1:

            down1= z.reshape(z.shape[0], z.shape[1],z.shape[2]//2,2, z.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1)

            if self.drop_initial:
                a,b,c,d = down1.shape
                e=d//self.pacth_num
                down1=down1.reshape(a,b,c,self.pacth_num,e)
                down1=down1.permute(0,1,3,2,4)
                down1=down1.reshape(a,b,down1.size()[2],-1)

                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'norm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                down1 = self.W_P_2(down1)
                if self.linear==0:
                    down1 = self.W_P_22(down1)

                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                z=down1
            else:
                a,b,c,d = z.shape
                e=d//self.pacth_num
                z=z.reshape(a,b,c,self.pacth_num,e)
                z=z.permute(0,1,3,2,4)
                z=z.reshape(a,b,z.size()[2],-1)                                           # x: [bs x nvars x patch_num x patch_len]

                if self.centralization: 
                    z = rearrange(z, "x y l c -> (x l) c y")
                    z = self.revin3(z, 'norm')
                    z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)
                z = self.W_P_1(z)
                if self.linear==0:
                    z = self.W_P_11(z)
                if self.centralization: 
                    z = rearrange(z, "x y l c -> (x l) c y")
                    z = self.revin3(z, 'denorm')
                    z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)

                a,b,c,d = down1.shape
                e=d//self.pacth_num
                down1=down1.reshape(a,b,c,self.pacth_num,e)
                down1=down1.permute(0,1,3,2,4)
                down1=down1.reshape(a,b,down1.size()[2],-1)
                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'norm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                down1 = self.W_P_2(down1)
                if self.linear==0:
                    down1 = self.W_P_22(down1)                
                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                z=0.5*(z+down1)
        elif self.multiscale==2:

            down1= z.reshape(z.shape[0], z.shape[1],z.shape[2]//2,2, z.shape[3])
            down1= down1.sum(dim=-2) 
            down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
            down1= down1.mean(dim=-1)

            down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
            down2= down2.sum(dim=-2) 
            down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
            down2= down2.mean(dim=-1) 


            if self.drop_initial:

                a,b,c,d = down1.shape
                e=d//self.pacth_num
                down1=down1.reshape(a,b,c,self.pacth_num,e)
                down1=down1.permute(0,1,3,2,4)
                down1=down1.reshape(a,b,down1.size()[2],-1)

                
                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'norm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)

                down1 = self.W_P_2(down1)
                if self.linear==0:
                    down1 = self.W_P_22(down1)    

                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'denorm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)


                a,b,c,d = down2.shape
                e=d//self.pacth_num
                down2=down2.reshape(a,b,c,self.pacth_num,e)
                down2=down2.permute(0,1,3,2,4)
                down2=down2.reshape(a,b,down2.size()[2],-1)

                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'norm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 
                down2 = self.W_P_3(down2)
                if self.linear==0:
                    down2 = self.W_P_33(down2)    
                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'denorm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 

                z=0.5*(down1+down2)

            else:
                a,b,c,d = z.shape
                e=d//self.pacth_num
                z=z.reshape(a,b,c,self.pacth_num,e)
                z=z.permute(0,1,3,2,4)
                z=z.reshape(a,b,z.size()[2],-1)                                           # x: [bs x nvars x patch_num x patch_len]

                if self.centralization: 
                    z = rearrange(z, "x y l c -> (x l) c y")
                    z = self.revin3(z, 'norm')
                    z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)

                z = self.W_P_1(z)
                if self.linear==0:
                    z = self.W_P_11(z)                    
                if self.centralization: 
                    z = rearrange(z, "x y l c -> (x l) c y")
                    z = self.revin3(z, 'denorm')
                    z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)


                a,b,c,d = down1.shape
                e=d//self.pacth_num
                down1=down1.reshape(a,b,c,self.pacth_num,e)
                down1=down1.permute(0,1,3,2,4)
                down1=down1.reshape(a,b,down1.size()[2],-1)

                if self.centralization:    
                    down1 = rearrange(down1, "x y l c -> (x l) c y")
                    down1 = self.revin1(down1, 'norm')
                    down1 = rearrange(down1, "(x l) c y -> x y l c",l=self.pacth_num)
                down1 = self.W_P_2(down1)
                if self.linear==0:
                    down1 = self.W_P_22(down1)  
                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'denorm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 

                a,b,c,d = down2.shape
                e=d//self.pacth_num
                down2=down2.reshape(a,b,c,self.pacth_num,e)
                down2=down2.permute(0,1,3,2,4)
                down2=down2.reshape(a,b,down2.size()[2],-1)

                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'norm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 
                down2 = self.W_P_3(down2)
                if self.linear==0:
                    down2 = self.W_P_33(down2)                  
                if self.centralization:   
                    down2 = rearrange(down2, "x y l c -> (x l) c y")
                    down2 = self.revin2(down2, 'denorm')
                    down2 = rearrange(down2, "(x l) c y -> x y l c",l=self.pacth_num) 

                z=(z+down1+down2)*(1/3)

        else:

            a,b,c,d = z.shape
            e=d//self.pacth_num
            z=z.reshape(a,b,c,self.pacth_num,e)
            z=z.permute(0,1,3,2,4)
            z=z.reshape(a,b,z.size()[2],-1)                                           # x: [bs x nvars x patch_num x patch_len]

            if self.centralization: 
                z = rearrange(z, "x y l c -> (x l) c y")
                z = self.revin3(z, 'norm')
                z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)

            z = self.W_P_1(z)
            if self.linear==0:
                z = self.W_P_11(z) 
            if self.centralization: 
                z = rearrange(z, "x y l c -> (x l) c y")
                z = self.revin3(z, 'denorm')
                z = rearrange(z, "(x l) c y -> x y l c",l=self.pacth_num)


        z =self.backbone(z)
        z =self.head(z)

        return z


class Interaction_backbone(nn.Module):
    def __init__(self,configs, context_window, target_window,cut1 ,cut2, d_model2,dropout2,n_heads,n_layers):
        super().__init__()

        self.cut1=cut1
        self.cut2=cut2
        self.context_window=context_window
        self.target_window=target_window

        # self.patch=configs.patch
        self.channel_mask=configs.channel_mask

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout2,
                                      output_attention=1), d_model2, n_heads),
                    d_model2,
                    d_model2,
                    dropout=dropout2,
                    activation='gelu'
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model2)
        )

        self.flatten = nn.Flatten(start_dim=-2)

        self.patch_num=configs.patch_num
        patch_len=int((context_window*(context_window//2))/self.patch_num)

        # if self.patch==1:
        #     patch=int(context_window//self.patch_num)
        #     self.num_patch=self.cut1//patch
        #     dim=d_model2//self.num_patch
        #     self.linear = nn.Linear(patch_len,dim)
        #     self.linear2= nn.Linear(dim*self.num_patch, d_model2)
        # else:
        self.linear=nn.Linear(cut1*(context_window//2),d_model2)


        if self.cut2 <self.target_window:
            self.proj=nn.Linear(d_model2,cut2 )
        else:
            self.proj=nn.Linear(d_model2,self.target_window )

        # self.cluster = Linear_extractor_cluster(configs)
        self.revin = RevIN(configs.enc_in)


        
    def forward(self, z,channel_mask):

        z=z[:,:,1:,:]         
        # if self.cut1==self.context_window:
        #     z=self.linear(self.flatten(z))
        # else:
        #     z=self.linear(self.flatten(z[:,:,:,-self.cut1:]))

        # if self.patch==1:

        #     a,b,c,d = z.shape
        #     e=d//self.patch_num
        #     z=z.reshape(a,b,c,self.patch_num,e)
        #     z=z.permute(0,1,3,2,4)
        #     z=z.reshape(a,b,z.size()[2],-1)                                           # x: [bs x nvars x patch_num x patch_len]

        #     z = rearrange(z, "x y l c -> (x l) c y")
        #     z = self.revin(z, 'norm')
        #     z = rearrange(z, "(x l) c y -> x y l c",l=self.patch_num)


        #     z = self.linear(z)

        #     z = rearrange(z, "x y l c -> (x l) c y")
        #     z = self.revin(z, 'denorm')
        #     z = rearrange(z, "(x l) c y -> x y l c",l=self.patch_num)

        #     z=self.linear2(self.flatten(z[:,:,-self.num_patch:,:]))

        # else:

        z=self.flatten(z[:,:,:,-self.cut1:])

        if self.cut1<self.context_window:
            z=z.permute(0,2,1)
            z = self.revin(z, 'norm')
            z=z.permute(0,2,1)

        z=self.linear(z)
        
        if self.cut1<self.context_window:

            z=z.permute(0,2,1)
            z = self.revin(z, 'denorm')
            z=z.permute(0,2,1)
            
        # channel_independent_input = rearrange(z, 'b l n -> (b l) n 1')

        # reshaped_output =self.cluster(channel_independent_input)

        # temporal_feature = rearrange(reshaped_output, '(b l) 1 n -> b l n', b=z.size()[0])

        if self.channel_mask:
            z,attention=self.encoder(z,attn_mask=channel_mask)
        else:
            z,attention=self.encoder(z)

        z=self.proj(z)

        if self.cut2 <self.target_window:
            a,b,d=z.size()
            zeros=torch.zeros((a,b,self.target_window), device=z.device)
            zeros[:,:,:self.cut2]=z
            return zeros
        else:
            return z



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        self.pos_enc=1
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model=d_model
        
        # Input encoding
        q_len = patch_num

        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
                                                          # x: [bs x nvars x patch_num x d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  
        # Encoder
        z = self.encoder(u) # z: [bs * nvars x patch_num x d_model]

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z  



class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            # list_log=[]
            for mod in self.layers: 
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # list_log.append(logits)
            return output
        else:
            # list_log=[]
            for mod in self.layers: 
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # list_log.append(logits)
            return output




class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))


        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head 
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward

        src2 =self.ff(src)
        # src2 = self.ff(src2)
        # src2,logits2= self.moe2(src2)
        # logits=torch.cat([logits1.unsqueeze(0),logits2.unsqueeze(0)],axis=0)

        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights



class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights

        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights




class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)


    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x=self.linear(x)
        return x


