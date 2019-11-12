import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from miscc.config import cfg
import numpy as np


class GLU(nn.Module):
    def __init__(self):
        """
        """

        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Pixel_Norm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        """

        super(Pixel_Norm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp


class Instance_Norm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """ 
        """

        super(Instance_Norm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=(2, 3), keepdim=True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, dim=(2, 3), keepdim=True) + self.epsilon)
        return x * tmp


class Apply_Noise(nn.Module):
    def __init__(self, channels):
        """
        """

        super(Apply_Noise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class Apply_Style(nn.Module):
    def __init__(self,
                 channels,      # Channels
                 resolution,    # Resolution
                 use_attn = True
                ):
        """
        """

        super(Apply_Style, self).__init__()
        self.w_dim = cfg.W_DIM
        self.a_dim = cfg.A_DIM
        self.channels = channels
        self.resolution = resolution
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = GlobalAttentionGeneral(channels, resolution)
            self.fc = nn.Linear(self.w_dim + self.a_dim, self.channels*2)
        else:
            self.fc = nn.Linear(self.w_dim, self.channels*2)
        
    def forward(self, x, w_code, word_embedding):
        """
        Inputs:
            x: [batch_size, channels, resolution, resolution]
            w_code: [batch_size, cfg.W_DIM], latent(W)
            word_embedding: [batch_size, 768, max_length], word embedding
        Outputs:
            x: [batch_size, channels, resolution, resolution]
        """

        if self.use_attn:
            attn_code = self.attn(x, word_embedding)
            style_code = torch.cat((attn_code, w_code), dim=1)
            style_code = self.fc(style_code) # [batch_size, n_channels*2]
        else:
            style_code = self.fc(w_code)     # [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1), 1, 1]
        style_code = style_code.view(shape) 
        x = x * (style_code[:, 0] + 1.) + style_code[:, 1]
        
        return x

class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
        """

        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x




# class FC(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  gain=2**(0.5),
#                  use_wscale=False,
#                  lrmul=1.0,
#                  bias=True):
#         """
#             The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
#         """
#         super(FC, self).__init__()
#         he_std = gain * in_channels ** (-0.5)  # He init
#         if use_wscale:
#             init_std = 1.0 / lrmul
#             self.w_lrmul = he_std * lrmul
#         else:
#             init_std = he_std / lrmul
#             self.w_lrmul = lrmul

#         self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#             self.b_lrmul = lrmul
#         else:
#             self.bias = None

#     def forward(self, x):
#         if self.bias is not None:
#             out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
#         else:
#             out = F.linear(x, self.weight * self.w_lrmul)
#         out = F.leaky_relu(out, 0.2, inplace=True)
#         return out
    



# class Conv2d(nn.Module):
#     def __init__(self,
#                  input_channels,
#                  output_channels,
#                  kernel_size,
#                  gain=2 ** (0.5),
#                  use_wscale=False,
#                  lrmul=1,
#                  bias=True):
#         '''
#         '''
#         super(Conv2d, self).__init__()
#         he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
#         self.kernel_size = kernel_size
#         if use_wscale:
#             init_std = 1.0 / lrmul
#             self.w_lrmul = he_std * lrmul
#         else:
#             init_std = he_std / lrmul
#             self.w_lrmul = lrmul

#         self.weight = nn.Parameter(
#             torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(output_channels))
#             self.b_lrmul = lrmul
#         else:
#             self.bias = None

#     def forward(self, x):
#         if self.bias is not None:
#             return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
#         else:
#             return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


###########################################
class GlobalAttentionGeneral(nn.Module):
    def __init__(self, channels, res):
        super(GlobalAttentionGeneral, self).__init__()
        self.cdf = cfg.WORD_DIM
        self.idf = channels
        self.ih = res
        self.iw = res
        self.a_dim = cfg.A_DIM
        self.conv_context = nn.Conv2d(self.cdf, self.idf, kernel_size=1, stride=1, bias=False)
        self.sm = nn.Softmax(dim = 1)
        self.mask = None
        self.conv = nn.Conv2d(self.idf, 1, kernel_size=1, stride=1, bias=False)
        self.att_fc = nn.Linear(self.ih*self.iw, self.a_dim, bias=True)

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, inputs, context):
        """
        Inputs:
            inputs: batch x idf x ih x iw (queryL=ihxiw)   image(h) idf = deepth
            context: batch x cdf x sourceL                 word embedding sequence, cdf = 768, sourceL = 18
        """
        ih, iw = inputs.size(2), inputs.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)


        target = inputs.view(batch_size, -1, queryL)
        # --> batch x queryL x idf
        targetT = torch.transpose(target, 1, 2).contiguous()

        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)

        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        #           h                       e'
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)

        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        # --> batch x (idf * queryL)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weightedContext = self.conv(weightedContext)
        weightedContext = weightedContext.view(batch_size, -1)
        weightedContext = self.att_fc(weightedContext)

        return weightedContext