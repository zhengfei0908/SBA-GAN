import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from miscc.config import cfg as config
import numpy as np


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
    

class Pixel_Norm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Pixel_Norm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1
def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

class Apply_Style(nn.Module):
    def __init__(self,
                 channels,      # Channels
                 res,           # Resolution
                 use_attn = True,
                 use_wscale = True,
                 **kwargs
                ):
        super(Apply_Style, self).__init__()
        self.w_dim = config.W_DIM
        self.a_dim = config.A_DIM
        self.channels = channels
        self.res = res
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = GlobalAttentionGeneral(channels, res)
            self.fc = FC(self.w_dim + self.a_dim,
                         self.channels*2,
                         gain=1.0,
                         use_wscale = use_wscale
                        )
        else:
            self.fc = FC(self.w_dim,
                         self.channels*2,
                         gain=1.0,
                         use_wscale = use_wscale
                        )
        
    def forward(self, x, w_code, word_embedding):
        '''
        Inputs:
            x: [batch_size, num_features, height, width], outputs of last synthesis process
            w_code: [batch_size, config.W_DIM], latent(W)
            word_embedding: [batch_size, max_length, 768], word embedding
        Outputs:
            x: [batch_size, num_features, height, width]
        '''
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
class Apply_Noise(nn.Module):
    def __init__(self, channels):
        '''
        '''
        super(Apply_Noise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        '''
        '''
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
            x = nn.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x

class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        '''
        '''
        super(Conv2d, self).__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)

class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        """

        """
        super(Upscale2d, self).__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x

class Instance_Norm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """ 
        """
        super(Instance_Norm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


###########################################
class GlobalAttentionGeneral(nn.Module):
    def __init__(self, channels, res):
        super(GlobalAttentionGeneral, self).__init__()
        self.cdf = config.E_DIM
        self.idf = channels
        self.ih = res
        self.iw = res
        self.a_dim = config.A_DIM
        self.conv_context = conv1x1(self.cdf, self.idf)
        self.sm = nn.Softmax(dim = 1)
        self.mask = None
        self.conv = nn.Conv2d(self.idf, 1, kernel_size=(1,1), bias=True)
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


### BERT_EMBEDDING
class BERT_EMBEDDING(nn.Module):
    def __init__(self):
        super(BERT_EMBEDDING, self).__init__()
        self.max_length = config.TEXT.MAX_LENGTH
        self.pretrained_model = config.TEXT.PRETRAINED_MODEL
        self.model = BertModel.from_pretrained(self.pretrained_model)
    
    def forward(self, indexed_tokens):
        '''
        Inputs:
            indexed_tokens: [batch_size, max_length]
            
        Outputs:
            words_embs: [batch_size, max_length, 768], word embedding
            sent_emb: [batch_size, 768], 
        '''
        input_ids = indexed_tokens
        segment_ids = torch.tensor([0] * self.max_length)
        mask_ids = input_ids != 0
        if config.CUDA:
            segment_ids = segment_ids.to('cuda')
            mask_ids = mask_ids.to('cuda')
        
        words_embs, sent_emb = self.model(input_ids, segment_ids, mask_ids, output_all_encoded_layers=False)
        words_embs = torch.transpose(words_embs, 1, 2)
        
        return words_embs, sent_emb
    
#     def _make_inputs(self, indexed_tokens):
#         '''Convert tokenized text to three inputs as Bert required
        
#         indexed_tokens: list of tokenized test, shape[batch_size, none]
#         '''
        
#         l = len(indexed_tokens)
#         # max_len = min(bucket_len, self.max_length)
#         mask_ids = indexed_tokens != 0
#         segment_ids = torch.tensor([0] * self.max_length)
        
#         return input_ids, segment_ids, mask_ids

### CA_NET
class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.e_dim = config.E_DIM
        self.c_dim = config.C_DIM
        self.fc = nn.Linear(self.e_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        log_var = x[:, self.c_dim:]
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        if config.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, sent_embs):
        '''
        Inputs:
            sent_embs: [batch_size, config.E_DIM=768]
            
        Outputs:
            c_code: [batch_size, config.C_DIM], reparametrized text embedding
            mu: [batch_size, config.C_DIM], mean
            log_var: [batch_size, config.C_DIM], logVariance
        '''
        
        mu, log_var = self.encode(sent_embs)
        c_code = self.reparametrize(mu, log_var)
        return c_code, mu, log_var

### MAPPING
class G_MAPPING(nn.Module):
    def __init__(self,
                 normalize_latents=True, # Normalize latent vector?
                 use_wscale=True,        # Enable equalized learning rate?
                 lrmul=0.01,             # Learning rate multiplier for the mapping layers
                 gain=2**(0.5),          # Original gain in tensorflow.
                 **kwargs
                 ):
        super(G_MAPPING, self).__init__()
        self.num_layers = config.M.LAYERS
        self.c_dim = config.C_DIM
        self.z_dim = config.Z_DIM
        self.w_dim = config.W_DIM
        self.concat_dim = self.c_dim + self.z_dim
        self.normalize_latents = normalize_latents
        if normalize_latents:
            self.pixel_norm = Pixel_Norm()
        else:
            self.pixel_norm = None
        
        self.mapping = nn.ModuleList()
        for idx in range(self.num_layers):
            if idx == 0:
                self.mapping.append(FC(self.concat_dim, self.w_dim, gain, lrmul, use_wscale))
            else:
                self.mapping.append(FC(self.w_dim, self.w_dim, gain, lrmul, use_wscale))

    def forward(self, c_code, z_code):
        '''
        Inputs:
            c_code: [batch_size, config.M.CONDITION_DIM+config.M.LATENT_DIM], text after CA_NET
            z_code: [batch_size, config.M.MAPPING_DIM], noise(Z) generated from some distribution
            
        Outputs:
            w_code: [batch_size, config.M.MAPPING_DIM], latent(W)
        '''
        if self.normalize_latents:
            z_code = self.pixel_norm(z_code)
        w_code = torch.cat((c_code, z_code), dim=1)
        for fc in self.mapping:
            w_code = fc(w_code)
        return w_code

class Layer_Epilogue(nn.Module):
    def __init__(self,
                 channels,
                 res,
                 use_attn = True,
                 use_wscale = True,
                 use_noise = True,
                 use_pixel_norm = False,
                 use_instance_norm = True,
                 **kwargs
                 ):
        super(Layer_Epilogue, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_noise:
            self.noise = Apply_Noise(channels)
        
        if use_pixel_norm:
            self.pixel_norm = Pixel_Norm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = Instance_Norm()
        else:
            self.instance_norm = None

        self.style_mod = Apply_Style(channels, res, **kwargs)

    def forward(self, x, w_code, word_embedding, noise=None):
        '''
        Inputs:
            x: [batch_size, channels, res, res]
            w_code: [batch_size, config.M.MAPPING_DIM], latent(W)
            word_embedding: [batch_size, max_length, 768], word embedding
            noise:
        Outputs:
            x: [batch_size, num_features, height, width]
        '''
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        x = self.style_mod(x, w_code, word_embedding)

        return x

class G_BLOCK(nn.Module):
    def __init__(self,
                 log2_res,           # Current Resolution,  3.4...log_2(resolution)
                 use_attn = True,
                 use_wscale = True,
                 use_noise = True,
                 use_pixel_norm = False,
                 use_instance_norm = True,
                 f=None,             # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=4096,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=256,       # Maximum number of feature maps in any layer.
                 **kwargs
                 ):
        super(G_BLOCK, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)
        

        # res
        self.log2_res = log2_res
        
        self.channel = self.nf(self.log2_res)
        
        # blur2d
        self.blur = Blur2d(f)

        # noise
        # self.noise = noise

        if self.nf(self.log2_res-1) == self.channel:
            # upsample method 1, 
            self.up_sample = Upscale2d(factor)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(self.log2_res-1), self.channel, 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = Layer_Epilogue(self.channel,
                                     2 ** self.log2_res,
                                     use_attn = use_attn,
                                     use_wscale = use_wscale,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )
        self.conv1  = Conv2d(self.channel, self.channel,
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = Layer_Epilogue(self.channel,
                                     2 ** self.log2_res,
                                     use_attn = use_attn,
                                     use_wscale = use_wscale,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )

    def forward(self, x, w_code, word_embedding, noise=None):
        '''
        Inputs:
            x: [batch_size, channels, res, res]
            w_code: [batch_size, config.M.MAPPING_DIM], latent(W)
            word_embedding: [batch_size, max_length, 768], word embedding
            noise:
        Outputs:
            x: [batch_size, num_features, height, width]
        '''
        x = self.up_sample(x)
        x = self.blur(x)
        x = self.adaIn1(x, w_code, word_embedding, noise)
        x = self.conv1(x)
        x = self.adaIn2(x, w_code, word_embedding, noise)
        return x

class G_NET(nn.Module):
    def __init__(self,
                 fmap_base = 4096,         # Overall multiplier for the number of feature maps
                 out_channels = 3,         # Number of output image colors
                 structure = 'fixed',      # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient
                 fmap_max = 256,           # Maximum inumber of feature maps in any layer
                 fmap_decay = 1.0,         # log2 feature map reduction when doubling the resolution
                 f=None,                   # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_attn = True,          # Enable attention style?
                 use_pixel_norm = False,   # Enable pixelwise feature vector normalization?
                 use_instance_norm = True, # Enable instance normalization?
                 use_wscale = True,        # Enable equalized learning rate?
                 use_noise = True,         # Enable noise inputs?
                 **kwargs
                ):
        super(G_NET, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)  # number of features
        self.structure = structure
        self.resolution_log2 = int(np.log2(config.RESOLUTION))
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.bert_embedding = BERT_EMBEDDING()
        for param in self.bert_embedding.parameters():
            param.requires_grad = False
#         param_count(self.bert_embedding)
        
        self.ca_net = CA_NET()
#         param_count(self.ca_net)
        
        self.mapping = G_MAPPING()
#         param_count(self.mapping)
        
        # noise input
#         self.noise_inputs = []
#         for layer_idx in range(num_layers):
#             res = layer_idx // 2 + 2
#             shape = [1, 1, 2 ** res, 2 ** res]
#             if config.CUDA:
#                 self.noise_inputs.append(torch.randn(*shape).to('cuda'))
#             else:
#                 self.noise_inputs.append(torch.randn(*shape))

    
        ## first layer
        self.x = nn.Parameter(torch.ones(1, self.nf(2), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(2)))
        self.adaIn1 = Layer_Epilogue(self.nf(2),
                                     4,
                                     use_attn = use_attn,
                                     use_wscale = use_wscale,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )
#         param_count(self.adaIn1)
        self.conv1  = Conv2d(self.nf(2), self.nf(2),
                             kernel_size=3, use_wscale=use_wscale)
        
#         param_count(self.conv1)
        self.adaIn2 = Layer_Epilogue(self.nf(2),
                                     4,
                                     use_attn = use_attn,
                                     use_wscale = use_wscale,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )
#         param_count(self.adaIn2)
        
        ## remaining layers
        self.generator = nn.ModuleList()
        for log2_res in range(3, self.resolution_log2+1):
            self.generator.append(G_BLOCK(log2_res))
        
        ## to image
        self.torgb = Conv2d(self.nf(self.resolution_log2), out_channels, kernel_size=1, gain=1, use_wscale=use_wscale)
        
    def forward(self, text, z_code):
        '''
        Inputs:
        
        Outputs:
            
        '''
        words_embs, sent_emb = self.bert_embedding(text)
        c_code, _, _ = self.ca_net(sent_emb)
        w_code = self.mapping(c_code, z_code)
        x = self.x.expand(z_code.size(0), -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.act(x)
        x = self.adaIn1(x, w_code, words_embs)
        x = self.conv1(x)
        x = self.adaIn2(x, w_code, words_embs)
        
        for block in self.generator:
            x = block(x, w_code, words_embs)
        x = self.torgb(x)
        
        return x, words_embs, sent_emb

class D_BLOCK(nn.Module):
    def __init__(self,
                 log2_res,           # Current Resolution,  log_2(resolution)...3
                 f=None,             # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=4096,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=256,       # Maximum number of feature maps in any layer.
                 **kwargs
                 ):
        super(D_BLOCK, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.blur2d = Blur2d(f)
        self.log2_res = log2_res
        
        self.channel = self.nf(self.log2_res)
        
        if self.nf(self.log2_res-1) == self.channel:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = nn.Conv2d(self.channel, self.nf(self.log2_res-1), kernel_size=2, stride=2)
        
        self.conv = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=(1,1))
        
    def forward(self, x):
        '''
        '''
        x = self.act(self.conv(x))
        x = self.act(self.downsample(self.blur2d(x)))
        
        return x

class D_GET_LOGITS(nn.Module):
    def __init__(self, img_dim, embedding_dim, condition=True):
        super(D_GET_LOGITS, self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        
        ## condition
        if self.condition:
            self.intermediate = nn.Sequential(
                nn.Conv2d(self.img_dim + self.embedding_dim, self.img_dim, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2)
            )

        self.logits_out = nn.Sequential(
            nn.Conv2d(self.img_dim, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )
        
    def forward(self, h_code, c_code=None):
        '''
        Inputs:
            h_code: image features after D_NET
            c_code: sentence embedding after CA_NET
        Outputs:
        
        '''
        if self.condition and c_code is not None:
            c_code = c_code.view(-1, self.embedding_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), dim=1)
            h_c_code = self.intermediate(h_c_code)
        else:
            h_c_code = h_code
            
        output = self.logits_out(h_c_code).view(-1)
        return output

class D_NET(nn.Module):
    def __init__(self,
                 condition = True,       # Use uncondition?
                 fmap_base = 4096,         # Overall multiplier for the number of feature maps
                 out_channels = 3,         # Number of output image colors
                 structure = 'fixed',      # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient
                 fmap_max = 256,           # Maximum inumber of feature maps in any layer
                 fmap_decay = 1.0,         # log2 feature map reduction when doubling the resolution
                 f=None,                   # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 **kwargs
                ):
        super(D_NET, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)  # number of features
        self.structure = structure
        self.condition = condition
        self.resolution_log2 = int(np.log2(config.RESOLUTION))
        self.act = nn.LeakyReLU(negative_slope=0.2)
        
        self.fromrgb = nn.Conv2d(out_channels, self.nf(self.resolution_log2), kernel_size=1)
        
        self.blur2d = Blur2d(f)
        
        self.discriminator = nn.ModuleList()
        for log2_res in range(self.resolution_log2, 2, -1):
            self.discriminator.append(D_BLOCK(log2_res))
        
        self.img_dim = self.nf(2)
        self.embedding_dim = config.E_DIM

        if self.condition:
            self.cond_dnet = D_GET_LOGITS(self.img_dim, self.embedding_dim, condition = True)
        else:
            self.cond_dnet = None
        self.uncond_dnet = D_GET_LOGITS(self.img_dim, self.embedding_dim, condition = False)
        
#         self.conv_last = nn.Conv2d(self.nf(2), self.nf(1), kernel_size=3, padding=(1, 1))
#         self.fc1 = nn.Linear(fmap_base, int(fmap_base / 4))
#         self.fc2 = nn.Linear(int(fmap_base / 4), 1)
        
#         self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        '''
        '''
        x = self.act(self.fromrgb(x))
        for block in self.discriminator:
            x = block(x)
#         x = self.act(self.conv_last(x))
#         x = x.view(x.size(0), -1)
#         x = self.act(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
        ## return features 512 x 4 x 4
        return x

class IMAGE_ENCODER(nn.Module):
    def __init__(self):
        super(IMAGE_ENCODER, self).__init__()
        self.img_dim = config.E_DIM
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.upsample = nn.Upsample(size=(299, 299))
        self.model.fc = nn.Linear(2048, self.img_dim)
        self.model.AuxLogits.fc = nn.Linear(768, self.img_dim)

    def forward(self, x):
        x = self.upsample(x)
        x = self.model(x)
        return x[0]