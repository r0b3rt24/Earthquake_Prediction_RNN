import torch
from Dataloader import Data 
from Model import GRU 
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time



print("==========> Preparing data")
raw_data = pd.read_csv('../data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
batch_size = 1024
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("training on GPU")
else:
    device = torch.device("cpu")
    print("training on CPU")

mask = np.random.rand(len(raw_data)) < 0.8
train = raw_data[mask]
test = raw_data[~mask]

train_set = Data(train)
test_set = Data(test)

train_loader = DataLoader(train_set)
test_loader = DataLoader(test_set)

def train(train_loader, lr = 0.01, hidden_dim = 256, epochs = 5):
    input_size = next(iter(train_loader))[0].shape[2]
    output_size = 1
    n_layers = 2

    model = GRU(input_size, hidden_dim, output_size, n_layers)

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_times = []

    print('=========> Starting training')
    for epoch in range(1, epochs + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0

        for x, label in train_loader:
            counter += 1
            h = h.data
        model.zero_grad()

        out, h = model(x.to(device).float(), h)

        loss = criterion(out, label.to(device).float())

        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if counter % 100 == 0:
            print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))

        end_time = time.clock()

        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, epochs, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(start_time-start_time)))
        epoch_times.append(start_time-start_time)
        
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    MAE = 0
    for i in range(len(outputs)):
        MAE += np.mean(abs(outputs[i]-targets[i])/len(outputs))
        
    print("MAE: {}%".format(MAE*100))
    return outputs, targets, MAE

model = train(train_loader)
    