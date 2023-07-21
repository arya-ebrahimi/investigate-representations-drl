from core.nn import InputReconstruction, Network
from torchsummary import summary
import torch

device = 'cuda'

model1 = Network(True, 'ir').to(device=device)

t = torch.rand((128, 3, 15, 15)).to(device=device)
print(t.shape)
x, aux = model1(t)


summary(model1, (3, 15, 15), 128, device=device)

# t = torch.rand((1024)).to(device=device)
# model = InputReconstruction().to(device)
# model(t)

# # summary(model, (1024), 128, device=device)