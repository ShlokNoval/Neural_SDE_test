# utils.py

import pandas as pd
import torch
import matplotlib.pyplot as plt

def load_data(file_path):   
    df = pd.read_excel(file_path, header=None) #reads the data from the excel file
    x = torch.tensor(df[0].values, dtype=torch.float32).view(-1, 1) #converts the data to a tensor where -1 is the number of rows and 1 is the number of columns
    t = torch.tensor(df[1].values, dtype=torch.float32) #converts the data to a tensor
    return x, t #returns the data

def plot_learned_physics(model, x_range):
    device = next(model.parameters()).device
    with torch.no_grad():   #no_grad is used to turn off gradient computation
        x_vals = torch.linspace(*x_range, 200).view(-1, 1).to(device) #creates 200 points between x_range where *x_range is the start and end of the range
        g1, g2 = model(x_vals) #predicts the drift and diffusion
    plt.plot(x_vals.cpu(), g1.cpu(), label='g1(x) - Drift') #plots the drift
    plt.plot(x_vals.cpu(), g2.cpu(), label='g2(x) - Diffusion') #plots the diffusion
    plt.xlabel('x') #x-axis label
    plt.ylabel('g(x)') #y-axis label
    plt.legend() #legend
    plt.title('Learned Hidden Physics')
    plt.show()
