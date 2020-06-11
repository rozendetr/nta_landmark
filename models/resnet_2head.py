from .resnet import *
from .mish import BetaMish
import torch.nn as nn
import torch

model_resnet = resnet34()
expansion = 1


class ResNet_2head(nn.Module):
    def __init__(self, expansion=expansion, num_classes_1=10, num_classes_2=10):
        super(ResNet_2head, self).__init__()
        self.expansion = expansion
        self.share_model = nn.Sequential(*list(model_resnet.children())[:-1])

        # nn.Sequential(*list(model_resnet.children())[-1]).in
        self.relu = nn.ReLU(inplace=True)
        self.mish = BetaMish(beta=1.5)


        self.fc1 = nn.Linear(512 * self.expansion, 256 * self.expansion)
        self.bn1 = nn.BatchNorm1d(256 * self.expansion, eps=2e-1)

        #heads
        self.h1 = nn.Linear(256 * self.expansion, num_classes_1)
        self.h2 = nn.Linear(256 * self.expansion, num_classes_2)

    def forward(self, x):
        x = self.share_model(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.mish(x)
        x = self.bn1(x)

        y1 = self.h1(x)
        y2 = self.h2(x)

        return y1, y2

    def freeze_shared(self):
        """
        Freeze shared net
        :return:
        """
        for param in self.share_model.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        """
        Unfreeze shared net
        :return:
        """
        for param in self.share_model.parameters():
            param.requires_grad = True


class ResNet_Mhead(nn.Module):
    def __init__(self, m_heads, expansion=1):
        """
        :param m_heads: dictionary {key_head: num_classes}
        :param expansion:
        """
        super(ResNet_2head, self).__init__()
        self.expansion = expansion
        self.m_heads = m_heads
        self.share_model = nn.Sequential(*list(model_resnet.children())[:-1])

        # nn.Sequential(*list(model_resnet.children())[-1]).in
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512 * self.expansion, 256 * self.expansion)
        self.bn1 = nn.BatchNorm1d(256 * self.expansion, eps=2e-1)

        #heads
        self.heads = {}
        for key in self.m_heads.keys():
            self.heads[key] = nn.Linear(256 * self.expansion, self.m_heads[key])

    def forward(self, x):
        x = self.share_model(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        y = []
        for key in self.m_heads.keys():
            y.append(self.heads[key](x))
        return y

    def freeze_shared(self):
        """
        Freeze shared net
        :return:
        """
        for param in self.share_model.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        """
        Unfreeze shared net
        :return:
        """
        for param in self.share_model.parameters():
            param.requires_grad = True


