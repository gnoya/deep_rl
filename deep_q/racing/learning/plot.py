import torch
import matplotlib.pyplot as plt
from parameters import *
from argparse import ArgumentParser
import os
import numpy as np

plt.rcParams["figure.figsize"] = (11.8,7.5)
# print(plt.rcParams["figure.figsize"])

parser = ArgumentParser()
parser.add_argument('--instance', required=True)
parser.add_argument('--detailed')
parser.add_argument('--both')
parser.add_argument('--win_rate')
args = parser.parse_args()
dir = '../instances/{}'.format(args.instance)
assert os.path.exists(dir) , 'Resume directory not found'

if args.detailed is not None:
    for i in range(4):
        model_file = os.path.join(dir, 'agent_model_' + str(i))
        model_file = os.path.join(model_file, 'checkpoint.pth.tar')
        assert os.path.exists(model_file), 'Error: model file not fund in folder'
        if(torch.cuda.is_available()):
            model_data = torch.load(model_file)
        else:
            model_data = torch.load(model_file, map_location='cpu')
        plt.plot(model_data['elo_history'], label="agent {} elo".format(i))

else:
    # Plot first team
    model_file = os.path.join(dir, 'agent_model_0')
    model_file = os.path.join(model_file, 'checkpoint.pth.tar')
    assert os.path.exists(model_file), 'Error: model file not found in folder'

    if(torch.cuda.is_available()):
        model_data = torch.load(model_file)
    else:
        model_data = torch.load(model_file, map_location='cpu')
    
    if(args.win_rate is not None):
        plot_variable = model_data['win_rate_history']
        plot_label = "Team A's win rate (agents 0 and 2)"
    else:
        plot_variable = model_data['elo_history']
        plot_label = "Team A's elo (agents 0 and 2)"
        
    plt.plot(plot_variable, label=plot_label)
    plot_np = np.array(plot_variable)
    mean = np.mean(plot_np)

    horiz_line_data = np.array([mean for i in range(len(plot_variable))])
    plt.plot(horiz_line_data, 'black', label='Average 0: {0:.2f}'.format(mean)) 

    if args.both is not None:
        # Plot second team
        model_file = os.path.join(dir, 'agent_model_1')
        model_file = os.path.join(model_file, 'checkpoint.pth.tar')
        assert os.path.exists(model_file), 'Error: model file not found in folder'

        if(torch.cuda.is_available()):
            model_data = torch.load(model_file)
        else:
            model_data = torch.load(model_file, map_location='cpu')

        if(args.win_rate is not None):
            plot_variable = model_data['win_rate_history']
            plot_label = "Team B's win rate (agents 1 and 3)"
        else:
            plot_variable = model_data['elo_history']
            plot_label = "Team B's elo (agents 1 and 3)"

        plt.plot(plot_variable, label=plot_label)
        plot_np = np.array(plot_variable)
        mean = np.mean(plot_np)

        horiz_line_data = np.array([mean for i in range(len(plot_variable))])
        plt.plot(horiz_line_data, 'black', label='Average 1: {0:.2f}'.format(mean))

legend = plt.legend(loc='upper center', shadow=True)
plt.title('Agents elo')
plt.ylabel('Elo')
plt.xlabel('Episodes (x{})'.format(saving_model_freq))
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()