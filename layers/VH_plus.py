from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class BinaryConcrete(nn.Module):
    def __init__(self, temp,batch_size,ndim):
        super(BinaryConcrete, self).__init__()
        # self.gumbel = torch.distributions.Gumbel(
        #     torch.zeros([batch_size,ndim,1]), torch.ones([batch_size,ndim,1]))
        self.temp = temp
        self.sigmoid = nn.Sigmoid()

    def forward(self,alpha):
        noise = torch.rand_like(alpha).cuda()
        noise=torch.log(noise)-torch.log(1-noise)
        ouput=self.sigmoid((alpha + noise) / self.temp)
        # ouput=self.sigmoid((alpha + self.gumbel.sample().cuda()) / self.temp)

        return ouput
    

class Concrete(nn.Module):
    def __init__(self, temp,batch_size,ndim):
        super(Concrete, self).__init__()
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros([batch_size,ndim,2]), torch.ones([batch_size,ndim,2]))  
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,alpha):
        return self.softmax((alpha + self.gumbel.sample().cuda()) / self.temp)
    
class Inference(nn.Module):
    def __init__(self, input=336, hidden=168, output=2,dim=7,individual=False):
        super(Inference, self).__init__()
        self.dim=dim
        self.individual=individual
        if individual:
            self.fc1 = nn.ModuleList()
            self.fc2 = nn.ModuleList()
            self.act_fn = nn.Tanh()
            for i in range(self.dim):
                self.fc1.append( nn.Linear(input, hidden))
                self.fc2.append(nn.Linear(hidden, output))

        else:
            self.fc1 = nn.Linear(input, hidden)
            self.fc2 = nn.Linear(hidden, output)
            self.sigmoid=nn.Sigmoid()
            self.act_fn = nn.Tanh()
        # self.fc1 = nn.Linear(input, hidden)
        # self.fc2 = nn.Linear(hidden, output)
        # self.act_fn = nn.Tanh()

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.dim):
                z = self.fc1[i](x[:,i,:])          # z: [bs x d_model * patch_num]
                z = self.act_fn(z)
                z = self.fc2[i](z)                    # z: [bs x target_window]
                x_out.append(z)
            h = torch.stack(x_out, dim=1) 
        else:
            h = self.fc1(x)
            h = self.act_fn(h)
            h = self.fc2(h)
            h=self.sigmoid(h)
            h_new=torch.log(h+1e-08*torch.ones_like(h)/(torch.ones_like(h)-h+1e-08*torch.ones_like(h)))

        return h_new,h


############Year_BASIS########################
class Year_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Year_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.05
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)
        self.weight = nn.Parameter(torch.zeros(dim))
        self.beta=beta
    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 


            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1

            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)

            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)
            
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):
        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()

    def vae_loss(self,alpha):
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  


        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[-1]).cuda()
        a,b= basis.size()


        for j in range(batch_size):
            index=x[j,2]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2+1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])

        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence


############WEEKDAY_BASIS########################

class Weekday_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Weekday_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.05
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)
        self.weight = nn.Parameter(torch.zeros(dim))
        self.beta=beta        
    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 


            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1
            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()

    def vae_loss(self,alpha):
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[0]).cuda()
        a,b= basis.size()

        for j in range(batch_size):
            index=x[j,0]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight


        # alpha=torch.ones_like(alpha,dtype=alpha.dtype)
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence

############WEEK_BASIS########################
class Week_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Week_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.2
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        zero = torch.zeros(7)
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)
        self.weight = nn.Parameter(torch.zeros(dim))
        self.beta=beta
    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 


            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1

            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()
       

    def vae_loss(self,alpha):
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[1]).cuda()
        a,b= basis.size()
        for j in range(batch_size):
            index=x[j,0]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence

############Day_BASIS########################
class Day_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Day_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.2
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)
        self.weight = nn.Parameter(torch.zeros(dim))
        self.beta=beta

    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 
            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1
            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)

            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()


    def vae_loss(self,alpha):
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl

    def forward(self, theta,x,y):  
        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[2]).cuda()
        a,b= basis.size()


        for j in range(batch_size):
            index=x[j,1]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2+1e-06*ones)
        alpha = self.encode(input)[0]

        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)

        return backcast, forecast, kl_diveregence


class VH(nn.Module):
    def __init__(self, configs):
        super(VH, self).__init__()
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        individual_embed=configs.individual_embed
        batch_size=configs.batch_size
        beta=configs.beta
        self.basis_list=[]
        self.embedding=configs.embedding
        if 0 in self.embedding:
            self.vae1=Year_basis(context_window,target_window,c_in,batch_size,individual_embed,beta)
            self.basis_list.append(self.vae1)
        if 1 in self.embedding:
            self.vae2=Weekday_basis(context_window,target_window,c_in,batch_size,individual_embed,beta)
            self.basis_list.append(self.vae2)
        if 2 in self.embedding:
            self.vae3=Week_basis(context_window,target_window,c_in,batch_size,individual_embed,beta)
            self.basis_list.append(self.vae3)
        if 3 in self.embedding:
            self.vae4=Day_basis(context_window,target_window,c_in,batch_size,individual_embed,beta)
            self.basis_list.append(self.vae4)

    def forward(self,x,y,z):


        adding=[]
        kl_divergence_total=[]
        for i in range(len(self.basis_list)):
            basis_back,basis_foward,kl_divergence=self.basis_list[i](x.permute(0,2,1),y,z)
            x=x-basis_back.permute(0,2,1)
            adding.append(basis_foward)
            kl_divergence_total.append(kl_divergence)

        x=x.detach()

        return x, adding, kl_divergence_total