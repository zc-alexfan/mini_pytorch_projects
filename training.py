import torch
import random
import numpy as np

def set_seed():
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.RandomState(1)
    torch.backends.cudnn.deterministic=True
    
def train(net, criterion, optimizer, trainloader, device, num_epoch, print_every, pbar): 
    set_seed()
    loss_vec = []
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        acc = running_loss/len(trainloader)
        print("[%d/%d] Loss = %.2f" % (epoch, num_epoch, acc))
        loss_vec.append(acc)

    print('Finished Training')
    return loss_vec