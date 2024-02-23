
def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    # 将31本来和为0，通过entropy变成数值
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def im(outputs_test, gent=True):
    # 16*31
    epsilon = 1e-10
    # 将16*31的输入中的每个31进行概率初始化
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    # 得到16*1，然后再求平均值
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        # 进行entropy，但是sum不一样
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss