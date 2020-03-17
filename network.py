import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


class Head2(torch.nn.Module):
    def __init__(self, inf_1, inf_2, out_f):
        super(Head2, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(inf_1+inf_2, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b11 = nn.BatchNorm1d(inf_1)
        self.b12 = nn.BatchNorm1d(inf_2)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.f(x1)
        x2 = self.f(x2)
        x1 = self.b11(x1)
        x2 = self.b12(x2)
        x = self.l(torch.cat((x1, x2), 1))
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)
        out = self.o(x)

        return out, x


class FCN2(torch.nn.Module):
    def __init__(self, base1, base2, inf_1, inf_2):
        super(FCN2, self).__init__()
        self.base1 = base1
        self.base2 = base2
        self.h1 = Head2(inf_1, inf_2, 1)

    def forward(self, x1, x2):
        x1 = self.base1(x1)
        x2 = self.base2(x2)
        return self.h1(x1, x2)


def Xception(in_f=2048):
    model = get_model("xception", pretrained=True)
    # model = get_model("resnet18", pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    # model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model = FCN(model, in_f)
    # model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    return model


def XceptionFusion(inf_1=2048, inf_2=2048):
    model_face = get_model("xception", pretrained=True)
    model_frame = get_model("xception", pretrained=True)
    model_face = nn.Sequential(*list(model_face.children())[:-1])
    model_frame = nn.Sequential(*list(model_frame.children())[:-1])
    model_face[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model_frame[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model = FCN2(model_face, model_frame, inf_1, inf_2)

    return model



