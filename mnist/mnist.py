import sys, os
import torch
from torchvision import datasets, transforms
import torchsnn
from sacred import Experiment
from sacred.observers import MongoObserver

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ex = Experiment('mnist_mike2018')


@ex.config
def config():
    batch_size = 5
    lr = 0.001
    total_its = 3
    
    arch = "stigmergic"
    
    n_inputs = 28
    n_outputs = 10
    time_ticks = 28

    if arch == "stigmergic":
        n_hidden = 10

    if arch == "lstm":
        n_layers = 3
        n_hidden = 10

    if arch == "recurrent":
        n_hidden = 28

    if arch == "feedforward":
        n_inputs = 28*28
        time_ticks = 1
        n_layers = 1
        n_hidden = 300

    avg_window = 100

    use_mongo = False
    if use_mongo:
        ex.observers.append(MongoObserver.create())

@ex.capture
def preProcess(x, n_inputs, time_ticks):
    x = torch.tensor(x, dtype=torch.double, device=device)
    x = x.reshape(-1, time_ticks, n_inputs)
    th = 0.05
    x[x >= th] = 1
    x[x < th] = 0
    return x


@ex.capture
def init_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/tmp/mnist_data", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/tmp/mnist_data", train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ])),batch_size=batch_size, shuffle=True)
    
    return (train_loader, test_loader)


@ex.capture
def getStigmergic(n_inputs, n_hidden, n_outputs ,time_ticks):
    from networks.stigmergic import StigmergicNetwork
    return StigmergicNetwork(
        torchsnn.SimpleLayer(n_inputs, n_hidden),
        torchsnn.FullLayer(n_hidden, n_hidden),
        torchsnn.TemporalAdapter(n_hidden, time_ticks),
        torch.nn.Linear(n_hidden*time_ticks, n_outputs),
        torch.nn.Sigmoid(),
        lastlayer=3,
    ).to(device)


@ex.capture
def getLSTM(n_inputs, n_hidden, n_layers, n_outputs):
    from networks.lstm import LSTMNetwork
    return LSTMNetwork(n_inputs,
        n_hidden,
        n_layers, 
        n_outputs,
    ).to(device)

@ex.capture
def getFF(n_inputs, n_hidden, n_layers, n_outputs):
    from networks.ff import FFNetwork
    return FFNetwork(
        n_inputs,
        n_hidden,
        n_layers, 
        n_outputs,
    ).to(device)


@ex.capture
def getRNN(n_inputs, n_hidden, n_outputs, batch_size):
    from networks.rnn import RNNNetwork
    return RNNNetwork(n_inputs,
        n_hidden,
        n_outputs,
        batch_size=batch_size
    ).to(device)


@ex.capture
def getNet(arch):
    if arch == "stigmergic":
        return getStigmergic()
    elif arch == "lstm":
        return getLSTM()
    elif arch == "feedforward":
        return getFF()
    elif arch == "recurrent":
        return getRNN()
    else:
        print("You can use one of the following architectures: 'stigmergic', 'feedforward', 'recurrent', 'lstm'")
        sys.exit(0)

@ex.capture
def train(net, train_loader, _run, lr, total_its, time_ticks, avg_window):
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    from utils import MovingAverage
    avg_loss = MovingAverage(avg_window)
    avg_accuracy = MovingAverage(avg_window)
    
    for it in range(0, total_its):
        for epoch, (batch_data, batch_target) in enumerate(train_loader):
            batch_out = net.forward_batch(preProcess(batch_data))
            
            loss = loss_fn(batch_out, batch_target.type(torch.long).to(batch_out.device))
            
            _, preds = batch_out.max(1)
            acc = (preds == batch_target.to(device)).float().mean()
            overall_epoch = len(train_loader)*it + epoch
            print("epoch:",overall_epoch,"/",len(train_loader)*total_its,"batch loss:",loss.item(), "batch accuracy: ",acc.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_accuracy += acc.item()

            _run.log_scalar("loss", float(avg_loss))
            _run.log_scalar("accuracy", float(avg_accuracy))

@ex.capture
def test(net, test_loader, batch_size):
    rights = 0.0
    tots = 0.0
    for batch_data, batch_target in test_loader:
        rights += net.evaluate_batch(preProcess(batch_data), batch_target)
        tots += batch_size
    return rights/tots


@ex.automain
def main(arch):
    train_loader, test_loader = init_loaders()
    net = getNet()
    train(net, train_loader)
    acc = test(net, test_loader)
    print("accuracy: ", acc)
    import pickle
    pickle.dump(net, open("results/"+arch+"_"+str(acc), "wb"))
    return acc