import torch

def my_custom_metric(pred, true):
    pred = pred[:, [6, 13, 27]]
    true = true[:, [6, 13, 27]]
    target = torch.where(true != 0)
    true = true[target]
    pred = pred[target]
    score = torch.mean(torch.abs((true - pred)) / (true))

    return score