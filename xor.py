import sys, os
import torch
import torchsnn

torch.manual_seed(3)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = torchsnn.Sequential(
    torchsnn.FullLayer(1,1,clamp_min=None,clamp_max=None)
).to(device)

dataset_X = [[[0], [0]],    [[0], [1]],    [[1], [0]],    [[1], [1]]]
dataset_Y = [    0,             1,             1,             0]

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

print("--- training ---")
accuracy = 0
i = 0
while accuracy < 1:
    i += 1
    loss = torch.tensor([0], dtype=torch.float32, device=device)
    accuracy = 0
    for X,Y in zip(dataset_X, dataset_Y):
        net.reset()
        out = None
        for xi in X:
            out = net(torch.tensor(xi, dtype=torch.float32, device=device))
            net.tick()
        loss += (Y-out[0][0])**2
        accuracy += 1.0 if Y == round(out[0][0].item()) else 0.0
    accuracy /= len(dataset_Y)
    loss.backward()
    optimizer.step()
    print("epoch: %d    loss: %f    accuracy: %f" % (i, loss.item(), accuracy))

print("--- testing ---")

for X,Y in zip(dataset_X, dataset_Y):
    net.reset()
    out = None
    for xi in X:
        out = net(torch.tensor(xi, dtype=torch.float32, device=device))
        net.tick()
    print("in: %s    out: %d (%f)" % (X, round(out.item()), out.item()))
