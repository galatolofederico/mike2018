import torch
import torchsnn
import numpy as np

torch.manual_seed(10)

import argparse

parser = argparse.ArgumentParser(description='Plot transfer functions of differnt layers')

parser.add_argument('-layer',
                    choices=["full", "simple", "feedforward"],
                    help='Layer to plot (default: full)',
                    default="full")

args = parser.parse_args()

layer = None
if args.layer == "full":
    layer = torchsnn.FullLayer(1,1,clamp_min=None,clamp_max=None)
elif args.layer == "simple":
    layer = torchsnn.SimpleLayer(1,1,clamp_min=None,clamp_max=None)
elif args.layer == "feedforward":
    layer = torch.nn.Sequential(
        torch.nn.Linear(1,1),
        torch.nn.Sigmoid()
    )

net = torchsnn.Sequential(
    layer
)

try:
    from matplotlib import pyplot as plt
except:
    print("In order to plot you need to install matplotlib")
    print("Run 'pip install -r plot_requirements.txt' inside your virtualenv")
    import sys
    sys.exit()

N = 10

start_x = -N
end_x = N

start_y = -N
end_y = N

step = 0.25

mat = np.zeros((int((end_x - start_x)/step), int((end_y-start_y)/step)))

for ii, i in enumerate(np.arange(start_x, end_x, step)):
    for jj, j in enumerate(np.arange(start_y, end_y, step)):
        net.reset()
        net(torch.tensor([i], dtype=torch.float32))
        net.tick()
        out = net(torch.tensor([j], dtype=torch.float32))
        mat[ii, jj] = out[0]


plt.matshow(mat)
plt.title(args.layer+" layer")
plt.show()
