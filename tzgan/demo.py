import torch
import matplotlib.pyplot as plt
import numpy as np
from generator import Generator


ngpu = 1
nz = 100
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = torch.load("C:\\Users\\Seb\\dev\\IA\\netG5epochCUDA")

noise = torch.randn(1, nz, 1, 1, device=device)
with torch.no_grad():
    newfake = netG(noise).detach().cpu()

plt.axis("off")
plt.imshow(np.transpose(newfake[0],(1,2,0)))
plt.show()