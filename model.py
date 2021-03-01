from torch import nn
from base import MNISTImageClassificationBase


class MNISTModel(MNISTImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.l4(x)
        return x