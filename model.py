import torch
import torch.nn as nn

class HiddenPhysicsNN(nn.Module): #pytorch module for neural network    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( #Feedforward Neural Network
            nn.Linear(1, 64), #input layer
            nn.Tanh(), #Tanh - activation function
            nn.Linear(64, 64), #hidden layer
            nn.Tanh(), #activation function
            nn.Linear(64, 2) #output layer
        )

    def forward(self, x): #forward pass - takes input x and returns output out
        out = self.net(x) #out is the output of the neural network
        g1 = out[:, [0]] #g1 is the first column of the output  #drift
        g2 = out[:, [1]] #g2 is the second column of the output  #diffusion
        return g1, torch.abs(g2)  # Ensure positive diffusion  #torch.abs(g2) is the absolute value of the diffusion

def drift(x, g1): #drift is the drift term of the SDE
    k = 1.0  # example known physics constant
    return -k * x + g1  #pull the particle back to equilibrium like a spring 

def diffusion(x, g2): #diffusion term of the SDE
    return g2  #g2 is the neural network’s prediction of how noise behaves at each point.