from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
from miscc.config import cfg
# from miscc.config import cfg
from miscc.utils import mkdir_p
# from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
# from model import G_DCGAN, G_NET
# from model import D_NET, G_NET,IMAGE_ENCODER,BERT_EMBEDDING(
from model import *
from datasets import prepare_data
from miscc.losses import sent_loss,g_loss,d_loss
# from miscc.losses import words_loss
# from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

#         torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.TEXT.PRETRAINED_MODEL)
    
    def build_models(self):
        # ###################encoders######################################## #
        
        image_encoder = IMAGE_ENCODER()
        print('Load image encoder incept v3')

        text_encoder = BERT_EMBEDDING()
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder bert')
        text_encoder.eval()

        # #######################generator and discriminators############## #
        epoch = 0
        netG = G_NET()
        netD = D_NET()
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            netD.cuda()
        # ########################################################### #
        

        return [text_encoder, image_encoder, netG, netD, epoch]
        # ########################################################### #
        
    def define_optimizers(self,netG, netD):
        optimizersD = optim.Adam(netD.parameters(),
                          lr=cfg.TRAIN.DISCRIMINATOR_LR,
                          betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)

        torch.save(netD.state_dict(),
                '%s/netD.pth' % (self.model_dir))
        print('Save G/Ds models.')

#     def set_requires_grad_value(self, models_list, brequires):
#         for i in range(len(models_list)):
#             for p in models_list[i].parameters():
#                 p.requires_grad = brequires

#     def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
#                          image_encoder, captions, cap_lens,
#                          gen_iterations, name='current'):
#         # Save images
#         fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
#         for i in range(len(attention_maps)):
#             if len(fake_imgs) > 1:
#                 img = fake_imgs[i + 1].detach().cpu()
#                 lr_img = fake_imgs[i].detach().cpu()
#             else:
#                 img = fake_imgs[0].detach().cpu()
#                 lr_img = None
#             attn_maps = attention_maps[i]
#             att_sze = attn_maps.size(2)
#             img_set, _ = \
#                 build_super_images(img, captions, self.ixtoword,
#                                    attn_maps, att_sze, lr_imgs=lr_img)
#             if img_set is not None:
#                 im = Image.fromarray(img_set)
#                 fullpath = '%s/G_%s_%d_%d.png'\
#                     % (self.image_dir, name, gen_iterations, i)
#                 im.save(fullpath)

#         # for i in range(len(netsD)):
#         i = -1
#         img = fake_imgs[i].detach()
#         region_features, _ = image_encoder(img)
#         att_sze = region_features.size(2)
#         _, _, att_maps = words_loss(region_features.detach(),
#                                     words_embs.detach(),
#                                     None, cap_lens,
#                                     None, self.batch_size)
#         img_set, _ = \
#             build_super_images(fake_imgs[i].detach().cpu(),
#                                captions, self.ixtoword, att_maps, att_sze)
#         if img_set is not None:
#             im = Image.fromarray(img_set)
#             fullpath = '%s/D_%s_%d.png'\
#                 % (self.image_dir, name, gen_iterations)
#             im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netD)
        print("Load optimizers")
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        z_code = torch.rand(batch_size, cfg.Z_DIM)
        
        if cfg.CUDA:
            z_code.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
#             while step < 3:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
#                 imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                real_imgs, text, cap_lens, _, _ = prepare_data(data)
                real_imgs = real_imgs[-1]
                #######################################################
                # (2) Generate fake images
                ######################################################
                
                fake_imgs, words_embs, sent_emb = netG(text,z_code)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                #######################################################
                # (3) Update D network
                ######################################################
                errD = 0
                D_logs = ''
                netD.zero_grad()
                errD = d_loss(netD, real_imgs, fake_imgs,sent_emb, real_labels, fake_labels)
                # backward and update parameters
                errD.backward()
                optimizersD.step()
                D_logs += 'errD: %.2f ' % (errD.data)

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1
                errG = 0
                G_logs = ''
                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG = g_loss(netD, image_encoder, fake_imgs, real_labels, sent_emb, match_labels)
                # backward and update parameters
                errG.backward()
                optimizerG.step()
                
                #not update G this time
#                 for p, avg_p in zip(netG.parameters(), avg_param_G):
#                     avg_p.mul_(0.999).add_(0.001, p.data)

                G_logs += 'errG: %.2f ' % (errG.data)
                
                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
#                 if gen_iterations  == 1:
                if gen_iterations % 1000 == 0:
                    self.save_singleimages(data,z_code,netG, '../output',
                          'test',gen_iterations)
#                     backup_para = copy_G_params(netG)
#                     load_params(netG, avg_param_G)
#                     self.save_img_results(netG, fixed_noise, sent_emb,
#                                           words_embs, mask, image_encoder,
#                                           captions, cap_lens, epoch, name='average')
#                     load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD.item(), errG.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 and epoch != 0:
                self.save_model(netG, avg_param_G, netD, epoch)

        self.save_model(netG, avg_param_G, netD, self.max_epoch)

    def save_singleimages(self,data,z_code,netG, save_dir,
                          split_dir,gen_iterations):

        real_imgs, text, _, _, keys = prepare_data(data)
    #         real_imgs = real_imgs[-1]
        fake_imgs, _, _ = netG(text,z_code)
        real_imgs = real_imgs[-1]
        for i in range(fake_imgs.size(0)):
            s_tmp = '%s/single_samples/%s/gen%s/' %\
                (save_dir, split_dir, gen_iterations//1000)
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            key = keys[i].replace('.','_')
            key = key.replace('/' , '_')
            tokens = '_'.join([self.tokenizer.convert_ids_to_tokens([w.item()])[0] for w in text[i]])
            fullpath_fake = '%s_%s_fake.jpg' % (s_tmp, tokens)
            fullpath_real = '%s_%s_real.jpg' % (s_tmp, tokens)


            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = fake_imgs[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath_fake)

            img2 = real_imgs[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img2.permute(1, 2, 0).data.cpu().numpy()
            im2 = Image.fromarray(ndarr)
            im2.save(fullpath_real)
        print('image saved')





#     def save_singleimages(self, images, filenames, save_dir,
#                           split_dir, sentenceID=0):
#         for i in range(images.size(0)):
#             s_tmp = '%s/single_samples/%s/%s' %\
#                 (save_dir, split_dir, filenames[i])
#             folder = s_tmp[:s_tmp.rfind('/')]
#             if not os.path.isdir(folder):
#                 print('Make a new folder: ', folder)
#                 mkdir_p(folder)

#             fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
#             # range from [-1, 1] to [0, 1]
#             # img = (images[i] + 1.0) / 2
#             img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
#             # range from [0, 1] to [0, 255]
#             ndarr = img.permute(1, 2, 0).data.cpu().numpy()
#             im = Image.fromarray(ndarr)
#             im.save(fullpath)

#     def sampling(self, split_dir):
#         if cfg.TRAIN.NET_G == '':
#             print('Error: the path for morels is not found!')
#         else:
#             if split_dir == 'test':
#                 split_dir = 'valid'
#             # Build and load the generator
#             if cfg.GAN.B_DCGAN:
#                 netG = G_DCGAN()
#             else:
#                 netG = G_NET()
#             netG.apply(weights_init)
#             netG.cuda()
#             netG.eval()
#             #
#             text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
#             state_dict = \
#                 torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
#             text_encoder.load_state_dict(state_dict)
#             print('Load text encoder from:', cfg.TRAIN.NET_E)
#             text_encoder = text_encoder.cuda()
#             text_encoder.eval()

#             batch_size = self.batch_size
#             nz = cfg.GAN.Z_DIM
#             noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
#             noise = noise.cuda()

#             model_dir = cfg.TRAIN.NET_G
#             state_dict = \
#                 torch.load(model_dir, map_location=lambda storage, loc: storage)
#             # state_dict = torch.load(cfg.TRAIN.NET_G)
#             netG.load_state_dict(state_dict)
#             print('Load G from: ', model_dir)

#             # the path to save generated images
#             s_tmp = model_dir[:model_dir.rfind('.pth')]
#             save_dir = '%s/%s' % (s_tmp, split_dir)
#             mkdir_p(save_dir)

#             cnt = 0

#             for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
#                 for step, data in enumerate(self.data_loader, 0):
#                     cnt += batch_size
#                     if step % 100 == 0:
#                         print('step: ', step)
#                     # if step > 50:
#                     #     break

#                     imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

#                     hidden = text_encoder.init_hidden(batch_size)
#                     # words_embs: batch_size x nef x seq_len
#                     # sent_emb: batch_size x nef
#                     words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
#                     words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
#                     mask = (captions == 0)
#                     num_words = words_embs.size(2)
#                     if mask.size(1) > num_words:
#                         mask = mask[:, :num_words]

#                     #######################################################
#                     # (2) Generate fake images
#                     ######################################################
#                     noise.data.normal_(0, 1)
#                     fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
#                     for j in range(batch_size):
#                         s_tmp = '%s/single/%s' % (save_dir, keys[j])
#                         folder = s_tmp[:s_tmp.rfind('/')]
#                         if not os.path.isdir(folder):
#                             print('Make a new folder: ', folder)
#                             mkdir_p(folder)
#                         k = -1
#                         # for k in range(len(fake_imgs)):
#                         im = fake_imgs[k][j].data.cpu().numpy()
#                         # [-1, 1] --> [0, 255]
#                         im = (im + 1.0) * 127.5
#                         im = im.astype(np.uint8)
#                         im = np.transpose(im, (1, 2, 0))
#                         im = Image.fromarray(im)
#                         fullpath = '%s_s%d.png' % (s_tmp, k)
#                         im.save(fullpath)

#     def gen_example(self, data_dic):
#         if cfg.TRAIN.NET_G == '':
#             print('Error: the path for morels is not found!')
#         else:
#             # Build and load the generator
#             text_encoder = \
#                 RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
#             state_dict = \
#                 torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
#             text_encoder.load_state_dict(state_dict)
#             print('Load text encoder from:', cfg.TRAIN.NET_E)
#             text_encoder = text_encoder.cuda()
#             text_encoder.eval()

#             # the path to save generated images
#             if cfg.GAN.B_DCGAN:
#                 netG = G_DCGAN()
#             else:
#                 netG = G_NET()
#             s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
#             model_dir = cfg.TRAIN.NET_G
#             state_dict = \
#                 torch.load(model_dir, map_location=lambda storage, loc: storage)
#             netG.load_state_dict(state_dict)
#             print('Load G from: ', model_dir)
#             netG.cuda()
#             netG.eval()
#             for key in data_dic:
#                 save_dir = '%s/%s' % (s_tmp, key)
#                 mkdir_p(save_dir)
#                 captions, cap_lens, sorted_indices = data_dic[key]

#                 batch_size = captions.shape[0]
#                 nz = cfg.GAN.Z_DIM
#                 captions = Variable(torch.from_numpy(captions), volatile=True)
#                 cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

#                 captions = captions.cuda()
#                 cap_lens = cap_lens.cuda()
#                 for i in range(1):  # 16
#                     noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
#                     noise = noise.cuda()
#                     #######################################################
#                     # (1) Extract text embeddings
#                     ######################################################
#                     hidden = text_encoder.init_hidden(batch_size)
#                     # words_embs: batch_size x nef x seq_len
#                     # sent_emb: batch_size x nef
#                     words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
#                     mask = (captions == 0)
#                     #######################################################
#                     # (2) Generate fake images
#                     ######################################################
#                     noise.data.normal_(0, 1)
#                     fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
#                     # G attention
#                     cap_lens_np = cap_lens.cpu().data.numpy()
#                     for j in range(batch_size):
#                         save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
#                         for k in range(len(fake_imgs)):
#                             im = fake_imgs[k][j].data.cpu().numpy()
#                             im = (im + 1.0) * 127.5
#                             im = im.astype(np.uint8)
#                             # print('im', im.shape)
#                             im = np.transpose(im, (1, 2, 0))
#                             # print('im', im.shape)
#                             im = Image.fromarray(im)
#                             fullpath = '%s_g%d.png' % (save_name, k)
#                             im.save(fullpath)

#                         for k in range(len(attention_maps)):
#                             if len(fake_imgs) > 1:
#                                 im = fake_imgs[k + 1].detach().cpu()
#                             else:
#                                 im = fake_imgs[0].detach().cpu()
#                             attn_maps = attention_maps[k]
#                             att_sze = attn_maps.size(2)
#                             img_set, sentences = \
#                                 build_super_images2(im[j].unsqueeze(0),
#                                                     captions[j].unsqueeze(0),
#                                                     [cap_lens_np[j]], self.ixtoword,
#                                                     [attn_maps[j]], att_sze)
#                             if img_set is not None:
#                                 im = Image.fromarray(img_set)
#                                 fullpath = '%s_a%d.png' % (save_name, k)
#                                 im.save(fullpath)