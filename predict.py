import torch
from model import MNISTModel

model = MNISTModel()
model.load_state_dict(torch.load('mnist_classification.pth'))

ip = torch.randn(1, 28, 28)

op = model.predict(torch.unsqueeze(ip, 0))

print(op)


