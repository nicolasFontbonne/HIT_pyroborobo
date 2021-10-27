from datetime import datetime
import matplotlib
import matplotlib.lines as mline
import matplotlib.pyplot as plt
print(matplotlib.matplotlib_fname())

import pandas as pd
import numpy as np
import csv
from os import listdir, mkdir


def distance(pos0x, pos0y, pos1x, pos1y):
    return ((pos1x-pos0x)**2+(pos1y-pos0y)**2)**0.5

folder = "logs/hit_2021-09-10-12-17-35-550016/"
infolder = listdir(folder)
if 'plots' not in infolder:
    mkdir(folder+'plots/')

position_x = []
with open(folder+"position_x.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        position_x.append([float(i) for i in row[:-1]])
        #plt.plot(amount_of_cooperate[-1])
position_x = np.array(position_x)


position_y = []
with open(folder+"position_y.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        position_y.append([float(i) for i in row[:-1]])
        #plt.plot(amount_of_cooperate[-1])
position_y = np.array(position_y)

n_agents = position_x.shape[1]







object_position_x = []
with open(folder+"object_position_x.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        object_position_x.append([float(i) for i in row[:-1]])
        #plt.plot(amount_of_cooperate[-1])
object_position_x = np.array(object_position_x)


object_position_y = []
with open(folder+"object_position_y.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        object_position_y.append([float(i) for i in row[:-1]])
        #plt.plot(amount_of_cooperate[-1])
object_position_y = np.array(object_position_y)

n_objects = object_position_x.shape[1]

object_radius = 10
robot_radius = 2.5

duration = 99
start_time = -100
# 119

k=11
list_k = [k]

list_k = range(n_agents)
list_k = [128, 124, 121, 120, 112, 84, 77, 66, 48, 42, 30, 15, 11]

for k in list_k:
    fig = plt.figure(figsize=(4,3), dpi=300) #figsize=(4,3)
    ax = fig.add_subplot(111)
    plt.xticks([])
    plt.yticks([])
    for i in range(n_agents):
        if i != k:
            plt.plot(position_x[start_time:start_time+duration,i], position_y[start_time:start_time+duration,i], 'k--', alpha=0.2)
            plt.plot(position_x[-1,i], position_y[-1,i], 'ko', markersize=robot_radius*2, alpha=0.2)

    for i in range(n_objects):
        draw_circle = plt.Circle((object_position_x[start_time,i], object_position_y[start_time,i]), 6,fc='#97D17B', alpha=0.3)
        ax.add_artist(draw_circle)
        draw_circle = plt.Circle((object_position_x[start_time,i], object_position_y[start_time,i]), object_radius, ec='#97D17B', fill=False, alpha=0.3, ls='--')
        ax.add_artist(draw_circle)
    for i in range(n_objects):
        draw_circle = plt.Circle((object_position_x[start_time+duration,i], object_position_y[start_time+duration,i]), 6,color='#97D17B')
        ax.add_artist(draw_circle)
        draw_circle = plt.Circle((object_position_x[start_time+duration,i], object_position_y[start_time+duration,i]),object_radius, ec='#97D17B',fill=False, ls='--')
        ax.add_artist(draw_circle)
    step_marker_list = []
    for step in range(start_time, start_time+duration):
        for i in range(n_agents):
            if i != k:
                if distance(position_x[step,k], position_y[step,k], position_x[step,i], position_y[step,i]) <= 16:
                    step_marker_list.append(step)
    #print(position_x[step_marker_list, k])
    #for i in range(n_objects):
    #    plt.plot(object_position_x[-1:,i], object_position_y[start_time:start_time+1,i], 'b+')
    plt.plot(position_x[start_time:start_time+duration,k], position_y[start_time:start_time+duration,k], 'k')
    plt.plot(position_x[start_time+duration,k], position_y[start_time+duration,k], 'ko', markersize=robot_radius*2)
    plt.plot(position_x[step_marker_list, k], position_y[step_marker_list, k], 'ro', markersize=2)
    line = plt.Line2D([np.amin(position_x)-robot_radius,np.amin(position_x)-robot_radius], [np.amin(position_y)-robot_radius,np.amax(position_y)+robot_radius])
    ax.add_line(line)

    line = plt.Line2D([np.amax(position_x)+robot_radius,np.amax(position_x)+robot_radius], [np.amin(position_y)-robot_radius,np.amax(position_y)+robot_radius])
    ax.add_line(line)

    line = plt.Line2D([np.amin(position_x)-robot_radius,np.amax(position_x)+robot_radius], [np.amin(position_y)-robot_radius,np.amin(position_y)-robot_radius])
    ax.add_line(line)

    line = plt.Line2D([np.amin(position_x)-robot_radius,np.amax(position_x)+robot_radius], [np.amax(position_y)+robot_radius,np.amax(position_y)+robot_radius])
    ax.add_line(line)

    xdim = 300
    ydim = int(xdim*3/4)
    plt.xlim([int(np.mean(position_x[start_time:start_time+duration,k])-xdim//2), int(np.mean(position_x[start_time:start_time+duration,k])+xdim//2)])
    plt.ylim([int(np.mean(position_y[start_time:start_time+duration,k])-ydim//2), int(np.mean(position_y[start_time:start_time+duration,k])+ydim//2)])

    #plt.xlim([int(np.amin(position_x[start_time:start_time+duration,k])-50), int(np.amax(position_x[start_time:start_time+duration,k])+50)])
    #plt.ylim([int(np.amin(position_y[start_time:start_time+duration,k])-100), int(np.amax(position_y[start_time:start_time+duration,k])+100)])
    plt.savefig(folder+f'plots/selection/trajectory_{k}_and_object.png')
    #plt.show()
    plt.close()
