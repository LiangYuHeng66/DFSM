import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_TAL(image_fetures, text_fetures, pid, tau, margin):

    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    mask = 1 - labels

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    scores = text_norm @ image_norm.t()

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
           + (-  (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)

    return loss.sum()



def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2
