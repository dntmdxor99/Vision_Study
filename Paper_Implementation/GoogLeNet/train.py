import torch
import arch
from torchsummary import summary


if torch.cuda.is_available():
    device = 'cuda'


# model = arch.Inception(192, 64, 96, 128, 16, 32, 32).to(device)
# x = torch.randn((1, 192, 244, 244)).to(device)
model = arch.auxClassifier(512, 1000).to(device)
x = torch.randn((1, 512, 14, 14)).to(device)
y = model(x)
print(y.shape)
# summary(model, input_size = (192, 244, 244), batch_size=64, device = device)

