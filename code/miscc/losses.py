import torch
import torch.nn as nn
from miscc.config import cfg

def sent_loss(sent_emb, img_code, match_labels, eps=1e-8):
    
    assert sent_emb.shape == img_code.shape
    if sent_emb.dim() == 2:
        sent_emb = sent_emb.unsqueeze(0)
        img_code = img_code.unsqueeze(0)
    
    ## [seq_len, batch_size, emb_dim]
    sent_emb_norm = torch.norm(sent_emb, 2, dim=2, keepdim=True)
    img_code_norm = torch.norm(img_code, 2, dim=2, keepdim=True)
    
    scores0 = torch.bmm(img_code_norm, sent_emb_norm.transpose(1, 2))
    norm0 = torch.bmm(img_code_norm, sent_emb_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.GAMMA3

    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)
    
    loss0 = nn.CrossEntropyLoss()(scores0, match_labels)
    loss1 = nn.CrossEntropyLoss()(scores1, match_labels)
    
    return loss0, loss1

def g_loss(d_net, img_encoder, fake_imgs, real_labels, sent_emb, match_labels):
    errG_total = 0
    
    features = d_net(fake_imgs)
    logits = d_net.uncond_dnet(features)
    errG = nn.BCELoss()(logits, real_labels)
    errG_total += errG
    
    if d_net.cond_dnet is not None:
        cond_logits = d_net.cond_dnet(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        errG_total += cond_errG
    
    img_code = img_encoder(fake_imgs)
    s_loss0, s_loss1 = sent_loss(sent_emb, img_code, match_labels)
    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.LAMBDA
    errG_total += s_loss
    
    return errG_total

def d_loss(d_net, real_imgs, fake_imgs, sent_emb, real_labels, fake_labels):
    
    real_features = d_net(real_imgs)
    fake_features = d_net(fake_imgs.detach())
    
    # uncond loss
    real_logits = d_net.uncond_dnet(real_features)
    fake_logits = d_net.uncond_dnet(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = real_errD + fake_errD
    
    if d_net.cond_dnet is not None:
        cond_real_logits = d_net.cond_dnet(real_features, sent_emb)
        cond_fake_logits = d_net.cond_dnet(fake_features, sent_emb)
        cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
        cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
        
        batch_size = real_features.size(0)
        cond_wrong_logits = d_net.cond_dnet(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
        cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])
        errD = ((real_errD + cond_real_errD) / 2.+
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    # minimize
    return errD