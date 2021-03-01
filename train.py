from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as t
from torchvision.datasets import MNIST
from model import MNISTModel
from device import to_device, get_default_device, DeviceDataLoader


def fit(model, epochs, lr, train_loader, valid_loader, opt=optim.Adam):
    history = []
    optimizer = opt(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_history = []
        for batch in train_loader:
            loss, result = model.training_step(batch)
            train_history.append(result)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_result = model.epoch_end(train_history)
        valid_result = model.evaluate(valid_loader)
        epoch_result = {
            'train_loss': train_result['loss'],
            'train_acc': train_result['acc'],
            'val_loss': valid_result['loss'],
            'val_acc': valid_result['acc']
        }
        model.epoch_end_log(epoch, epoch_result)
        history.append(epoch_result)
    return history


model = MNISTModel()
history = []

mnist_ds = MNIST('.', download=True, train=True, transform=t.ToTensor())
train_ds, valid_ds = random_split(mnist_ds, [55000, 5000])

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=64)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
model = to_device(model, device)


history += fit(model, 5, 1e-3, train_dl, valid_dl, opt=optim.Adam)

print(history)











