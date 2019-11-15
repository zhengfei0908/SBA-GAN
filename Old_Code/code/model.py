import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np

from miscc.config import cfg
from model_modules import GLU, Pixel_Norm, Instance_Norm, Apply_Noise, Apply_Style


### BERT_EMBEDDING
class BERT_EMBEDDING(nn.Module):
    def __init__(self):
        """
            Embedding layer
        """

        super(BERT_EMBEDDING, self).__init__()
        self.max_length = cfg.TEXT.MAX_LENGTH
        self.pretrained_model = cfg.TEXT.PRETRAINED_MODEL
        self.model = BertModel.from_pretrained(self.pretrained_model)
        self.e_dim = cfg.E_DIM
        self.fc = nn.Linear(768, self.e_dim, bias = True)
    
    def forward(self, indexed_tokens):
        """
        Inputs:
            indexed_tokens: [batch_size, max_length]
            
        Outputs:
            words_embs: [batch_size, 768, max_length]
            sent_emb: [batch_size, 768], 
        """

        input_ids = indexed_tokens
        segment_ids = torch.tensor([0] * self.max_length).to(input_ids.device)
        mask_ids = (input_ids != 0).to(input_ids.device)
        
        words_embs, sent_emb = self.model(input_ids, segment_ids, mask_ids, output_all_encoded_layers=False)
        words_embs = torch.transpose(words_embs, 1, 2)
        sent_emb = self.fc(sent_emb)

        return words_embs, sent_emb


### CA_NET
class CA_NET(nn.Module):
    def __init__(self):
        """
            Conditional augmentation
        """

        super(CA_NET, self).__init__()
        self.e_dim = cfg.E_DIM
        self.c_dim = cfg.C_DIM
        self.fc = nn.Linear(self.e_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        log_var = x[:, self.c_dim:]
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(0, 1)
        if cfg.CUDA:
            eps = eps.to('cuda')
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, sent_embs):
        """
        Inputs:
            sent_embs: [batch_size, cfg.E_DIM]
            
        Outputs:
            c_code: [batch_size, cfg.C_DIM], condition code after augmentation
            mu: [batch_size, cfg.C_DIM], mean
            log_var: [batch_size, cfg.C_DIM], logVariance
        """

        mu, log_var = self.encode(sent_embs)
        c_code = self.reparametrize(mu, log_var)
        return c_code, mu, log_var

### MAPPING
class G_MAPPING(nn.Module):
    def __init__(self):
        """
            Mapping layers
        """

        super(G_MAPPING, self).__init__()
        self.num_layers = cfg.M.LAYERS
        self.c_dim = cfg.C_DIM
        self.z_dim = cfg.Z_DIM
        self.w_dim = cfg.W_DIM
        self.concat_dim = self.c_dim + self.z_dim
        self.normalize = cfg.M.USE_NORM
        if self.normalize:
            self.pixel_norm = Pixel_Norm()
        else:
            self.pixel_norm = None
        
        self.mapping = nn.ModuleList()
        for idx in range(self.num_layers):
            if idx == 0:
                self.mapping.append(nn.Linear(self.concat_dim, self.w_dim))
            else:
                self.mapping.append(nn.Linear(self.w_dim, self.w_dim))

    def forward(self, c_code, z_code):
        """
        Inputs:
            c_code: [batch_size, cfg.C_DIM], text after augmentation
            z_code: [batch_size, cfg.W_DIM], noise(Z) generated from some distribution
            
        Outputs:
            w_code: [batch_size, cfg.W_DIM], latent(W)
        """

        if self.normalize:
            z_code = self.pixel_norm(z_code)
        w_code = torch.cat((c_code, z_code), dim=1)
        for fc in self.mapping:
            w_code = fc(w_code)
        return w_code

class Layer_Epilogue(nn.Module):
    def __init__(self,
                 channels,
                 resolution,
                 use_attn = True,
                 use_noise = True,
                 use_pixel_norm = False,
                 use_instance_norm = True
                 ):
        """
        """

        super(Layer_Epilogue, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.use_noise = use_noise
        
        if use_noise:
            self.apply_noise = Apply_Noise(channels)
        else:
            self.apply_noise = None
        
        if use_pixel_norm:
            self.pixel_norm = Pixel_Norm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = Instance_Norm()
        else:
            self.instance_norm = None

        self.apply_style = Apply_Style(channels, resolution, use_attn)

    def forward(self, x, w_code, word_embedding, noise=None):
        """
        Inputs:
            x: [batch_size, channels, res, res]
            w_code: [batch_size, cfg.W_DIM], latent(W)
            word_embedding: [batch_size, 768, max_length], word embedding
            noise:
        Outputs:
            x: [batch_size, channels, resolution, resolution]
        """
        if self.apply_noise is not None:
            x = self.apply_noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        x = self.apply_style(x, w_code, word_embedding)

        return x

class G_BLOCK(nn.Module):
    def __init__(self,
                 resolution,           # Current Resolution,  8,16,...,256
                 use_attn = True,
                 use_noise = True,
                 use_pixel_norm = False,
                 use_instance_norm = True,
                 factor=2,           # upsample factor.
                 fmap_base=4096,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=256,       # Maximum number of feature maps in any layer.
                 ):
        super(G_BLOCK, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)
        

        # res
        self.resolution_log2 = int(np.log2(resolution))
        assert 2 ** self.resolution_log2 == resolution
        
        self.channels = self.nf(self.resolution_log2)

        if self.nf(self.resolution_log2-1) == self.channels:
            # upsample method 1, 
            self.up_sample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(self.resolution_log2-1), self.channels, kernel_size=4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        # self.blur = Blur2d()
        self.adaIN1 = Layer_Epilogue(self.channels,
                                     resolution,
                                     use_attn = use_attn,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )
        self.conv  = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        self.adaIN2 = Layer_Epilogue(self.channels,
                                     resolution,
                                     use_attn = use_attn,
                                     use_noise = use_noise,
                                     use_pixel_norm = use_pixel_norm,
                                     use_instance_norm = use_instance_norm
                                    )

    def forward(self, x, w_code, word_embedding, noise=None):
        """
        Inputs:
            x: [batch_size, channels, resolution, resolution]
            w_code: [batch_size, 2, cfg.W_DIM], latent(W)
            word_embedding: [batch_size, 768, max_length], word embedding
            noise:
        Outputs:
            x: [batch_size, channels, resolution, resolution]
        """

        x = self.up_sample(x)
        # x = self.blur(x)
        x = self.adaIN1(x, w_code[:,0], word_embedding, noise)
        x = self.conv(x)
        x = self.adaIN2(x, w_code[:,1], word_embedding, noise)
        return x

class G_NET(nn.Module):
    def __init__(self,
                out_channels = 3,         # Number of output image colors
                fmap_base = 4096,         # Overall multiplier for the number of feature maps
                structure = 'fixed',      # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient
                factor = 2,
                fmap_max = 256,           # Maximum inumber of feature maps in any layer
                fmap_decay = 1.0,         # log2 feature map reduction when doubling the resolution
                use_attn = True,          # Enable attention style?
                use_noise = True,         # Enable noise inputs?
                use_pixel_norm = False,   # Enable pixelwise feature vector normalization?
                use_instance_norm = True, # Enable instance normalization?
                use_truncation = False,
                truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                ):
        """
        """

        super(G_NET, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)  # number of features
        self.structure = structure
        self.resolution = cfg.GAN.RESOLUTION
        self.resolution_log2 = int(np.log2(self.resolution))
        assert 2 ** self.resolution_log2 == self.resolution

        self.resolution_init = cfg.GAN.RESOLUTION_INIT
        self.resolution_init_log2 = int(np.log2(self.resolution_init))
        self.channels_init = self.nf(self.resolution_init_log2)
        
        self.num_layers = (self.resolution_log2 - self.resolution_init_log2 + 1) * 2

        self.use_attn = cfg.GAN.USE_ATTENTION
        self.use_noise = cfg.GAN.USE_NOISE
        self.use_pixel_norm = cfg.GAN.USE_PIXEL_NORM
        self.use_instance_norm = cfg.GAN.USE_INSTANCE_NORM
        self.use_truncation = cfg.GAN.USE_TRUNCATION
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.bert_embedding = BERT_EMBEDDING()
        for i, layer in enumerate(self.bert_embedding.children()):
            if i == 0:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.ca_net = CA_NET()

        self.mapping = G_MAPPING()
        
        # noise input
#         self.noise_inputs = []
#         for layer_idx in range(num_layers):
#             res = layer_idx // 2 + 2
#             shape = [1, 1, 2 ** res, 2 ** res]
#             if cfg.CUDA:
#                 self.noise_inputs.append(torch.randn(*shape).to('cuda'))
#             else:
#                 self.noise_inputs.append(torch.randn(*shape))

    
        ## first layer
        self.x = nn.Parameter(torch.ones(1, self.channels_init,
                                        self.resolution_init,
                                        self.resolution_init))
        self.bias = nn.Parameter(torch.ones(self.channels_init))

        self.adaIN1 = Layer_Epilogue(self.channels_init,
                                     self.resolution_init,
                                     use_attn = self.use_attn,
                                     use_noise = self.use_noise,
                                     use_pixel_norm = self.use_pixel_norm,
                                     use_instance_norm = self.use_instance_norm
                                    )
#         param_count(self.adaIn1)
        self.conv  = nn.Conv2d(self.channels_init, self.channels_init,
                                kernel_size=3, padding=1)
        
        self.adaIN2 = Layer_Epilogue(self.channels_init,
                                     self.resolution_init,
                                     use_attn = self.use_attn,
                                     use_noise = self.use_noise,
                                     use_pixel_norm = self.use_pixel_norm,
                                     use_instance_norm = self.use_instance_norm
                                    )
        
        ## remaining layers
        self.generator = nn.ModuleList()
        for log2_res in range(self.resolution_init_log2+1, self.resolution_log2+1):
            self.generator.append(G_BLOCK(2 ** log2_res,
                                        use_attn = self.use_attn,
                                        use_noise = self.use_noise,
                                        use_pixel_norm = self.use_pixel_norm,
                                        use_instance_norm = self.use_instance_norm,
                                        factor = factor,
                                        fmap_base = fmap_base,
                                        fmap_decay = fmap_decay,
                                        fmap_max = fmap_max
                                        ))

        ## to image
        self.torgb = nn.Conv2d(self.nf(self.resolution_log2), out_channels, kernel_size=1, stride=1)
        # self.tanh = nn.Tanh()

    def forward(self, text, z_code):
        ''' 
        '''

        words_embs, sent_emb = self.bert_embedding(text)
        c_code, mu, log_var = self.ca_net(sent_emb)

        w_code = self.mapping(c_code, z_code)
        w_code = w_code.unsqueeze(1)
        w_code = w_code.expand(-1, self.num_layers, -1)
        if self.use_truncation:
            coefs = np.ones([1, self.num_layers, 1], dtype=np.float32)
            for i in range(self.num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            w_code = w_code * (torch.Tensor(coefs).to(w_code.device))

        x = self.x.expand(z_code.size(0), -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.act(x)
        x = self.adaIN1(x, w_code[:,0], words_embs)
        x = self.conv(x)
        x = self.adaIN2(x, w_code[:,1], words_embs)
        
        for i, block in enumerate(self.generator):
            x = block(x, w_code[:,(i*2+2):(i*2+4)], words_embs)
        x = self.torgb(x)
        # x = self.tanh(x)
        
        return x, words_embs, sent_emb, mu, log_var


class D_BLOCK(nn.Module):
    def __init__(self,
                 resolution,           # Current Resolution,  log_2(resolution)...3
                 factor=2,           # upsample factor.
                 fmap_base=4096,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=256,       # Maximum number of feature maps in any layer.
                 ):
        """
        """

        super(D_BLOCK, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)
        

        self.act = nn.LeakyReLU(negative_slope=0.2)
        # self.blur2d = Blur2d()
        self.resolution_log2 = int(np.log2(resolution))
        
        self.channels = self.nf(self.resolution_log2)
        
        if self.nf(self.resolution_log2-1) == self.channels:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = nn.Conv2d(self.channels, self.nf(self.resolution_log2-1), kernel_size=2, stride=2)
        
        self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=(1,1))
        
    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.act(self.downsample(x))
        
        return x

class D_GET_OUTPUT(nn.Module):
    def __init__(self, img_dim, embedding_dim, condition=True):
        """
        """

        super(D_GET_OUTPUT, self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.wgan = cfg.LOSS.WGAN
        self.expand_num = cfg.GAN.RESOLUTION_INIT

        ## condition
        if self.condition:
            self.intermediate = nn.Sequential(
                nn.Conv2d(self.img_dim + self.embedding_dim, self.img_dim, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2)
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.img_dim, 1)
        self.logits = nn.Sigmoid()
        
    def forward(self, h_code, c_code=None):
        """
        Inputs:
            h_code: image features after D_NET
            c_code: sentence embedding after CA_NET
        Outputs:
        """

        if self.condition and c_code is not None:
            c_code = c_code.view(-1, self.embedding_dim, 1, 1)
            c_code = c_code.repeat(1, 1, self.expand_num, self.expand_num)
            h_c_code = torch.cat((h_code, c_code), dim=1)
            h_c_code = self.intermediate(h_c_code)
        else:
            h_c_code = h_code
        output = nn.AdaptiveAvgPool2d(1)(h_c_code)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        # print(output.shape)
        output = self.fc(output)
        # output = self.output(h_c_code).view(-1)
        if not self.wgan:
            output = self.logits(output)

        return output

class D_NET(nn.Module):
    def __init__(self,
                 condition = True,        # Use uncondition?
                 fmap_base = 4096,         # Overall multiplier for the number of feature maps
                 out_channels = 3,         # Number of output image colors
                 structure = 'fixed',      # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient
                 fmap_max = 256,           # Maximum inumber of feature maps in any layer
                 fmap_decay = 1.0,         # log2 feature map reduction when doubling the resolution
                ):
        super(D_NET, self).__init__()
        self.nf = lambda res: min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)  # number of features
        self.structure = structure
        self.condition = condition
        self.resolution_log2 = int(np.log2(cfg.GAN.RESOLUTION))
        self.resolution_init_log2 = int(np.log2(cfg.GAN.RESOLUTION_INIT))
        self.act = nn.LeakyReLU(negative_slope=0.2)
        
        self.fromrgb = nn.Conv2d(out_channels, self.nf(self.resolution_log2), kernel_size=1)
                
        self.discriminator = nn.ModuleList()
        for log2_res in range(self.resolution_log2, self.resolution_init_log2, -1):
            self.discriminator.append(D_BLOCK(2 ** log2_res))
        
        self.img_dim = self.nf(self.resolution_init_log2)
        self.embedding_dim = cfg.E_DIM

        if self.condition:
            self.cond_dnet = D_GET_OUTPUT(self.img_dim, self.embedding_dim, condition = True)
        else:
            self.cond_dnet = None
        self.uncond_dnet = D_GET_OUTPUT(self.img_dim, self.embedding_dim, condition = False)
        
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
        ## return features 256 x 8 x 8
        return x

class IMAGE_ENCODER(nn.Module):
    def __init__(self):
        super(IMAGE_ENCODER, self).__init__()
        self.img_dim = cfg.E_DIM
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