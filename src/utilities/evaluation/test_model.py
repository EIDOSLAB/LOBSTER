import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Evaluates model Top-k accuracy.
    :param output: PyTorch model's output.
    :param target: Inference ground truth.
    :param topk: Tuple defining the Top-k accuracy to be evaluated e.g. (1, 5) equals to Top-1 and Top-5 accuracy
    :return: List containing the Top-K accuracy values.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def test_model(model, loss_function, dataloader, device):
    """
    Evaluates PyTorch model performance.
    :param model: PyTorch model to evaluate.
    :param loss_function: Loss function used to evaluate the model Loss.
    :param dataloader: DataLoader on which evaluate the performance.
    :param device: Device on which to map the data, cpu or cuda:x where x is the cuda id.
    :return: Top-1 accuracy, Top-5 accuracy and Loss.
    """
    losses = AverageMeter('Loss')
    top1 = AverageMeter('@1Accuracy')
    top5 = AverageMeter('@5Accuracy')

    model.eval()

    if dataloader is not None:
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = loss_function(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

    return top1.avg, top5.avg, losses.avg
