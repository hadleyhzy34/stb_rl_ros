import torch
from matplotlib import pyplot as plt

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
lrs = []


for i in range(20):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    print("Factor = ", i, " , Learning Rate = ", optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(lrs)
plt.show()
