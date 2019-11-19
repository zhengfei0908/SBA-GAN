import argparse
import os
import random
import math
from scipy.stats import entropy

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.inception import inception_v3
from dataset import MultiResolutionDataset
# from datasets import TextDataset, prepare_data
from model import StyledGenerator, Discriminator, TextProcess


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.99):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=8):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD        

class Inception_score:
    def __init__(self, resize=True, splits=1):
        self.resize = resize
        self.splits = splits
        self.model = inception_v3(pretrained=True, transform_input=False).cuda()
        self.model.eval()
     
    def get_pred(self, x):
        if self.resize:
            up = nn.Upsample(size=(299, 299), mode='nearest').cuda()
            x = up(x)
        x = self.model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    def cal(self, imgs):
        N = imgs.size(0)
        imgs = imgs.cuda()
        preds = self.get_pred(imgs)
        split_scores = []

        for k in range(self.splits):
            part = preds[k * (N // self.splits): (k+1) * (N // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            
        return np.mean(split_scores)


def get_sentence(tokenizer, index):
    n = index.shape[0]
    batch_sent = []
    for i in range(n):
        batch_sent.append(' '.join([tokenizer.convert_ids_to_tokens([int(i)])[0] for i in index[i]]))
    return batch_sent
        
        
def train(args, dataset, text_process, generator, discriminator, inception_score):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.0001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.0002))

    pbar = tqdm(range(110000, 300_000))
    
    
    requires_grad(text_process, False)
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    kl_loss_val = 0
    l1_loss_val = 0
    score = 0
    
    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    # fixed output
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _, fixed_caption = next(data_loader)
    fixed_gen_i, fixed_gen_j = (10, 5)
    fixed_caption = fixed_caption.cuda()
    # fixed_c_code, _, _, _, _ = text_process(fixed_caption)
    descriptions = get_sentence(tokenizer,fixed_caption[:fixed_gen_j])
    
    with open(os.path.join(args.out, 'fixed_sentence.txt'), 'w') as f:
        for item in descriptions:
            f.write("%s\n" % item)
    
    # Training iteration
    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))
#         if (resolution == args.init_size and args.ckpt is None) or final_progress:
        if resolution == args.init_size or final_progress:
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
                    't_optimizer': t_optimizer.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                    't_running': t_running.state_dict(),
                },
                os.path.join(args.out, f'checkpoint/train_step-{ckpt_step}.model')
            )

            adjust_lr(t_optimizer, args.lr.get(resolution, 0.0001))
            adjust_lr(g_optimizer, args.lr.get(resolution, 0.0001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.0002))

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
            real_predict = discriminator(real_image, sent_emb, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward(retain_graph=True)
        
        ####Todo: fit for condition gan###
        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, sent_emb, step=step, alpha=alpha)
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
            gen_in1, gen_in2 = torch.randn(2, b_size, 384, device='cuda').chunk(
                2, 0
            )
            gen_in1, gen_in2 = gen_in1.squeeze(0), gen_in2.squeeze(0)
            gen_in1 = torch.cat([gen_in1, c_code], dim=1)
            gen_in2 = torch.cat([gen_in2, c_code], dim=1)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha)
        mismatch_predict = discriminator(real_image[:(b_size-1)], sent_emb[1:b_size], step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean() / 2.
            mismatch_predict = mismatch_predict.mean() / 2.
            (fake_predict + mismatch_predict).backward(retain_graph=True)

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, sent_emb, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if (i+1)%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict + mismatch_predict).item()

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
            
            if resolution <= 16:
                requires_grad(text_process, True)
            else:
                requires_grad(text_process.module.bert_embedding.fc, True)
                requires_grad(text_process.module.ca_net, True)
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, sent_emb, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = (-predict).mean()
                

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()
                
            kl_loss = KL_loss(mu, log_var)
            (loss + kl_loss).backward()
            
            if (i+1) % 10 == 0:
                gen_loss_val = loss.item()
                kl_loss_val = kl_loss.item()

            t_optimizer.step()
            g_optimizer.step()
            accumulate(t_running, text_process.module)
            accumulate(g_running, generator.module)
            
            requires_grad(text_process, False)
            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 2000 == 0:
            images = []
            with torch.no_grad():
                fixed_c_code, _, _, _, _ = t_running(fixed_caption)
                for _ in range(fixed_gen_i):
                    images.append(
                        g_running(
                            torch.cat([torch.randn(fixed_gen_j, 384).cuda(), fixed_c_code[:fixed_gen_j]], dim=1), step=step, alpha=alpha
                        ).data.cpu()
                    )
                    
            images = torch.cat(images, 0)
            score = inception_score.cal(images)
            
            utils.save_image(
                images,
                os.path.join(args.out, f'{str(i+1).zfill(6)}-{4 * 2 ** step}x{4 * 2 ** step}.png'),
                nrow=fixed_gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                t_running.state_dict(), os.path.join(args.out, f'checkpoint/t_{str(i + 1).zfill(6)}.model')
            )
            torch.save(
                g_running.state_dict(), os.path.join(args.out, f'checkpoint/g_{str(i + 1).zfill(6)}.model')
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.4f}; D: {disc_loss_val:.4f}; '
            f'KL: {kl_loss_val:.4f}; L1: {l1_loss_val}; Grad: {grad_loss_val:.4f}; '
            f'IS: {score: .4f}; '
            f'Alpha: {alpha:.4f};'
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
        default=640_000,
        help='number of samples used for each training phases',
    )
#     parser.add_argument('--local_rank', type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
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
    
#     torch.distributed.init_process_group(backend="nccl")
    text_process = nn.DataParallel(TextProcess(max_length=24, embedding_dim=128, condition_dim=128)).cuda()
    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    t_running = TextProcess(max_length=24, embedding_dim=128, condition_dim=128).cuda()
    
    t_running.train(False)
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)
    
    t_optimizer = optim.Adam(text_process.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer = optim.Adam(
        generator.module.generator.parameters(),
        lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2*args.lr, betas=(0.0, 0.99))
    
    accumulate(t_running, text_process.module, 0)
    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        
        text_process.module.load_state_dict(ckpt['text_process'])
        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        t_running.load_state_dict(ckpt['t_running'])
        g_running.load_state_dict(ckpt['g_running'])
        t_optimizer.load_state_dict(ckpt['t_optimizer'])
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
    
    dataset = MultiResolutionDataset(args.path, transform, max_length=24)
    inception_score = Inception_score(resize=True, splits=1)
    
    
    if args.sched:
        args.lr = {4: 1e-3, 8: 1e-3, 16: 1e-3, 32: 1e-4, 64: 1e-4, 128: 1e-4, 256: 1e-4}
        args.batch = {4: 128, 8: 128, 16: 64, 32: 32, 64: 32, 128: 16, 256: 16}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, text_process, generator, discriminator, inception_score)
