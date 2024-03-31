import torch


def accuracy(X, y):
    X, y = standard_Xy(X, y)
    return (torch.sum(X == y) / torch.numel(y)).item()

def f1(X, y):
    X, y = standard_Xy(X, y)
    epsilon = 1e-5

    TP = torch.sum(X * y).item()
    FP = torch.sum(X * (1 - y)).item()
    FN = torch.sum((1 - X) * y).item()

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    return 2 * recall * precision / (recall + precision + epsilon)

def standard_Xy(X, y):
    if X.shape[-3] != 1:
        X = torch.argmax(X, dim = -3)
    X = torch.squeeze(X)
    y = torch.squeeze(y)
    return X, y

def Confusion(X, y):
    X, y = standard_Xy(X, y)

    TP = torch.sum(X * y).item()
    FP = torch.sum(X * (1 - y)).item()
    TN = torch.sum((1-X) * (1-y)).item()
    FN = torch.sum((1 - X) * y).item()
    return TP, FP, TN, FN

def TP(X, y):
    return Confusion(X, y)[0]

def FP(X, y):
    return Confusion(X, y)[1]

def TN(X, y):
    return Confusion(X, y)[2]

def FN(X, y):
    return Confusion(X, y)[3]
    

