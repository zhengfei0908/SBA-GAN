from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.DATA_DIR = '../data/birds'
__C.GPU_ID = 0
__C.CUDA = False
__C.WORKERS = 4




__C.E_DIM = 256          # Sentence embedding dimension
__C.WORD_DIM = 768          # Word embedding dimension
__C.C_DIM = 128          # Condition c_code dimension
__C.Z_DIM = 128          # Random z_code dimension
__C.W_DIM = 256          # Latent w_code dimension
__C.A_DIM = 256          # Attention a_code dimension

# Text Embedding
__C.TEXT = edict()
__C.TEXT.PRETRAINED_MODEL = 'bert-base-uncased'
__C.TEXT.MAX_LENGTH = 18
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18


## Mapping
__C.M = edict()
__C.M.LAYERS = 8
__C.M.USE_NORM = True


## GAN
__C.GAN = edict()
__C.GAN.RESOLUTION_INIT = 8
__C.GAN.RESOLUTION = 256     # Target image's resolution
__C.GAN.USE_ATTENTION = True
__C.GAN.USE_NOISE = True
__C.GAN.USE_PIXEL_NORM = False
__C.GAN.USE_INSTANCE_NORM = True
__C.GAN.USE_TRUNCATION = True

__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


## LOSS
__C.LOSS = edict()
__C.LOSS.WGAN = True
__C.LOSS.WGAN_LAMBDA = 10


__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 8
__C.TRAIN.MAX_EPOCH = 100
__C.TRAIN.CRITIC_ITER = 5

__C.TRAIN.SNAPSHOT_INTERVAL = 10
__C.TRAIN.DISCRIMINATOR_LR = 1e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True


__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.TRAIN.LAMBDA = 0.5
__C.TRAIN.GAMMA3 = 1e-8

## Tree
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64




def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
