import torch
import torch.nn as nn
from model import HiddenPhysicsNN, drift, diffusion

def simulate_sde(x0, t, model):
    dt = float(t[1] - t[0]) #time steps
    steps = len(t) #total time steps
    paths = [x0] # Initialize with starting point   suppose 1/500
    x = x0

    for _ in range(steps - 1):
        g1, g2 = model(x) #predicts control values of NN
        dW = torch.randn_like(x) * dt**0.5      #Brownian Noise
        x = x + drift(x, g1) * dt + diffusion(x, g2) * dW   #Euler Maruyama    Why x + ? because using the previous x to predict the next x
        paths.append(x) #stores the predicted x values
    
    return torch.stack(paths)  

def train_model(x_data, t, device, epochs=600, batch_size=64, num_paths=64):   #batch size is the number of paths to simulate and num_paths is the number of paths to train on
    model = HiddenPhysicsNN().to(device) #HiddenPhysicsNN is the model we are training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #Adam is the optimizer we are using and lr=1e-3 is the learning rate means 0.001

    x0 = x_data[0].repeat(num_paths, 1).to(device) #64 simulated trajectories starting from same point.
    data_mean = x_data.mean() #mean of the data
    data_var = x_data.var() #variance of the data

    for epoch in range(epochs):
        optimizer.zero_grad() #zero the gradients
        sim_X = simulate_sde(x0, t, model) #simulate the SDE
        sim_mean = sim_X.mean(dim=1) #mean of the simulated data
        sim_var = sim_X.var(dim=1) #variance of the simulated data

        loss = ((sim_mean - data_mean)**2).mean() + ((sim_var - data_var)**2).mean()    #MSE loss
        loss.backward() #backpropagation - These gradients tell the optimizer how to change each parameter to reduce the loss.
        optimizer.step() #update the weights

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4e}")   #print the loss every 100 epochs

    return model
