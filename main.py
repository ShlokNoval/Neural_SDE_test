import torch
from utils import load_data, plot_learned_physics
from train import train_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x_data, t = load_data("data/water_X_trajectory_500fps_30kframes_0.02.xlsx")
    x_data, t = x_data.to(device), t.to(device) 

    model = train_model(x_data, t, device)  

    x_min, x_max = x_data.min().item(), x_data.max().item() #Gets the range of observed positions
    plot_learned_physics(model, (x_min, x_max))

if __name__ == "__main__":
    main()


