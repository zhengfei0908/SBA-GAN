import argparse
import os
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
# from datasets import TextDataset, prepare_data
from model import StyledGenerator, Discriminator, TextProcess

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=8):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, dataset, text_process, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(300_000), position=0, leave=True)
    
    
    requires_grad(text_process, False)
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_uncond_val = 0
    disc_loss_cond_val = 0
    disc_loss_mismatch_val = 0
    gen_loss_uncond_val = 0
    gen_loss_cond_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'text_process': text_process.module.state_dict(),
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                os.path.join(args.out, f'checkpoint/train_step-{ckpt_step}.model')
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image, caption = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, caption = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        caption = caption.cuda()
        c_code, sent_emb, words_embs, mu, log_var = text_process(caption)

        if args.loss == 'wgan-gp':
            real_predict, cond_real_predict = discriminator(real_image, sent_emb, step=step, alpha=alpha)
            #real_predict = real_predict.mean() / 2. - 0.001 * (real_predict ** 2).mean() / 2.
            cond_real_predict = cond_real_predict.mean() - 0.001 * (cond_real_predict ** 2).mean()
            #cond_real_predict = cond_real_predict / 100.
            
            #(-real_predict-cond_real_predict).backward(retain_graph=True)
            (-cond_real_predict).backward(retain_graph=True)
        
        ####Todo: fit for condition gan###
        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if (i+1)%10 == 0:
                grad_loss_val = grad_penalty.item()
                
        ####Todo: fit for condition gan###
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            ####Todo: hard code####
            gen_in1, gen_in2 = torch.randn(2, b_size, 512, device='cuda').chunk(
                2, 0
            )
            gen_in1, gen_in2 = gen_in1.squeeze(0), gen_in2.squeeze(0)
            #gen_in1 = torch.cat([gen_in1.squeeze(0), c_code], dim=1)
            #gen_in2 = torch.cat([gen_in2.squeeze(0), c_code], dim=1)

        fake_image = generator(gen_in1, c_code, step=step, alpha=alpha)
        fake_predict, cond_fake_predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha)
        #_, mismatch_predict = discriminator(real_image[:(b_size-1)], sent_emb[1:b_size], step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            #fake_predict = fake_predict.mean() / 3.
            cond_fake_predict = cond_fake_predict.mean()
            #cond_fake_predict = cond_fake_predict / 100.
            #mismatch_predict = mismatch_predict.mean() / 3.
            #mismatch_predict = mismatch_predict / 100.
            #(fake_predict+cond_fake_predict+mismatch_predict).backward(retain_graph=True)
            (cond_fake_predict).backward(retain_graph=True)

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict, cond_hat_predict = discriminator(x_hat, sent_emb, step=step, alpha=alpha)
#             grad_x_hat = grad(
#                 outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
#             )[0]
            cond_grad_x_hat = grad(
                outputs=cond_hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
#             grad_penalty = (
#                 (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
#             ).mean()
            cond_grad_penalty = (
                (cond_grad_x_hat.view(cond_grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
#             grad_penalty = 10 * grad_penalty
            cond_grad_penalty = 10 * cond_grad_penalty
#             grad_penalty.backward()
            cond_grad_penalty.backward()
            if (i+1)%10 == 0:
                grad_loss_val = cond_grad_penalty.item()
                #disc_loss_uncond_val = (-real_predict + fake_predict).item()
                disc_loss_cond_val = (-cond_real_predict + cond_fake_predict).item()
                #disc_loss_mismatch_val = mismatch_predict.item()
                #disc_loss_val = (-real_predict-cond_real_predict+fake_predict +cond_fake_predict).item() / 2.0

        ####Todo: fit for condition gan###
        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            text_process.zero_grad()
            generator.zero_grad()
            
            if resolution <= 2:
                requires_grad(text_process, True)
            else:
                requires_grad(text_process.module.bert_embedding.fc, True)
                requires_grad(text_process.module.ca_net, True)
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, c_code, step=step, alpha=alpha)

            predict, cond_predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                #uncond_loss = (-predict).mean() / 2.
                cond_loss = (-cond_predict).mean()
                #cond_loss = cond_loss / 100.
                #(uncond_loss + cond_loss).backward(retain_graph=True)
                (cond_loss).backward(retain_graph=True)

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            
            if (i+1) % 10 == 0:
                #gen_loss_uncond_val = uncond_loss.item()
                gen_loss_cond_val = cond_loss.item()

            g_optimizer.step()
            accumulate(g_running, generator.module)
            
            requires_grad(text_process, False)
            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 1000 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), c_code[:gen_j], step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                os.path.join(args.out, f'{str(i+1).zfill(6)}-{4 * 2 ** step}x{4 * 2 ** step}.png'),
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), os.path.join(args.out, f'checkpoint/{str(i + 1).zfill(6)}.model')
            )

        state_msg = (
            f'Size: {4 * 2 ** step};'
            f'G_u: {gen_loss_uncond_val:.3f}; G_c: {gen_loss_cond_val:.2f};'
            f'D_u: {disc_loss_uncond_val:.3f}; D_c: {disc_loss_cond_val:.2f}; D_m: {disc_loss_mismatch_val:.2f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.4f};'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=320_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=4, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument('--out', default=None, type=str, help='path of output folder')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()
    
    text_process = nn.DataParallel(TextProcess(max_length=18, embedding_dim=128, condition_dim=128)).cuda()
    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        list(generator.module.generator.parameters()) + list(text_process.parameters()),
        lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        
        text_process.module.load_state_dict(ckpt['text_process'])
        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if not os.path.exists(os.path.join(args.out, 'checkpoint')):
        os.makedirs(os.path.join(args.out, 'checkpoint'))
    
    ####Todo: choose a good max_length####
    dataset = MultiResolutionDataset(args.path, transform, max_length=18)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 64, 8: 64, 16: 32, 32: 32, 64: 32, 128: 16, 256: 16}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, text_process, generator, discriminator)
