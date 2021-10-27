import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir

def distance(pos0x, pos0y, pos1x, pos1y):
    return ((pos1x-pos0x)**2+(pos1y-pos0y)**2)**0.5

folder = "logs/2021-09-09-14-26-14-914404/"
fig = plt.figure()
exp_list = [s for s in listdir(folder) if '.pdf' not in s]

count_msg_evo_all = []

for exp in exp_list:
    exp_folder = folder+exp+'/'

    position_x = []
    with open(exp_folder+"position_x.csv") as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            position_x.append([float(i) for i in row[:-1]])
            #plt.plot(amount_of_cooperate[-1])
    position_x = np.array(position_x)


    position_y = []
    with open(exp_folder+"position_y.csv") as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            position_y.append([float(i) for i in row[:-1]])
            #plt.plot(amount_of_cooperate[-1])
    position_y = np.array(position_y)

    n_agents = position_x.shape[1]
    max_ite = position_x.shape[0]


    integration_time = 400




    count_msg_evo = []
    for start_time in range(0, max_ite, integration_time):
        count_msg = 0
        for k in range(n_agents):
            agent_com_list = []
            for step in range(start_time, min(max_ite, start_time+integration_time)):
                for i in range(n_agents):
                    if i != k:
                        if distance(position_x[step,k], position_y[step,k], position_x[step,i], position_y[step,i]) <= 16:
                            agent_com_list.append(i)
            count_msg += len(np.unique(agent_com_list))
        count_msg_evo.append(count_msg)
    count_msg_evo = np.array(count_msg_evo)
    count_msg_evo_all.append(np.sum(count_msg_evo, axis=1)[:-1]/count_msg_evo.shape[1])

count_msg_evo_all = np.array(count_msg_evo_all)
count_msg_evo_all_average = np.mean(count_msg_evo_all, axis=0)
count_msg_evo_all_std = np.std(count_msg_evo_all, axis=0)
iteration = list(range(0, 15000, 400))[:-1]

plt.figure(figsize=(4,3))
plt.plot(iteration, count_msg_evo_all_average, 'k', label='average')
plt.fill_between(iteration, count_msg_evo_all_average-count_msg_evo_all_std, 
                    count_msg_evo_all_average+count_msg_evo_all_std,alpha=0.2, color='k', label='standard deviation')
#plt.plot(sample_time, score)
plt.xlabel('iteration')
plt.ylabel('number of unique encounters \nper robot (average)')
plt.xlim([0, 14000])
plt.xticks([0, 14000//2, 14000])
plt.title('number of unique encounters per robot (average)')
plt.savefig(folder+'count_msg_evo_all.pdf')
plt.close()