import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# # Adapted from: https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):

        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
        target = target.view(-1,1)
        
        if self.ignore_index != None:
            # Filter predictions with ignore label from loss computation
            mask = target != self.ignore_index

            target = target[mask[:,0], :]
            input = input[mask[:,0], :]
        
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

