import matplotlib
print(matplotlib.matplotlib_fname())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir, mkdir
import sys

if len(sys.argv) <= 1:
    folder = "logs/2021-10-21-10-11-22-417759/hit_2021-10-21-11-14-45-969092/"
else:
    folder = sys.argv[1] +'/'
#folder = "logs/hit_no_T_2021-09-13-14-29-58-636373/"
print(folder)
infolder = listdir(folder)
if 'plots' not in infolder:
    mkdir(folder+'plots/')

max_iteration = 40000 #14000
# -------------------------------------------------------

fig = plt.figure(figsize=(4, 3))
score = []
sample_time = []
with open(folder+"sample_time.csv") as read_obj:
    csv_reader = list(csv.reader(read_obj))[0][:-1]
    #print(csv_reader)
    sample_time = [int(i) for i in csv_reader]

sample_time_parameters = []
with open(folder+"sample_time_parameters.csv") as read_obj:
    csv_reader = list(csv.reader(read_obj))[0][:-1]
    #print(csv_reader)
    sample_time_parameters = [int(i) for i in csv_reader]
    
with open(folder+"score.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        score.append(np.mean([float(i) for i in row]))
        #plt.plot(amount_of_cooperate[-1])
plt.plot(sample_time, score)
plt.xlabel('time step')
plt.ylabel('average reward')
plt.title('swarm performance wrt foraging')
plt.xlim([0, max_iteration])


plt.savefig(folder+'plots/score.pdf')
plt.close()

# -------------------------------------------------------

fig = plt.figure(figsize=(4, 3))
origin_parameter_0 = []
with open(folder+"origin_parameter_0.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        origin_parameter_0.append([float(i) for i in row])
        #plt.plot(amount_of_cooperate[-1])
origin_parameter_0 = np.array(origin_parameter_0)
plt.plot(sample_time_parameters, origin_parameter_0, 'k--')
plt.xlabel('time step')
plt.ylabel('origin of parameter 0')
plt.xlim([0, max_iteration])
plt.savefig(folder+'plots/origin_parameter_0.pdf')
plt.close()

fig = plt.figure(figsize=(4, 3))
lenrow = origin_parameter_0.shape[0]
count_parameter_0 = np.zeros(lenrow)
for i, row in enumerate(origin_parameter_0):
    count_parameter_0[i] = len(np.unique(row))
plt.plot(sample_time_parameters, count_parameter_0, 'k')
plt.xlabel('time step')
plt.ylabel('versions of parameter 0')
plt.xlim([0, max_iteration])
plt.savefig(folder+'plots/number_parameter_0.pdf')
plt.close()

# -------------------------------------------------------

fig = plt.figure(figsize=(4, 3))
value_parameter_0 = []
with open(folder+"value_parameters_0.csv") as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        value_parameter_0.append([float(i) for i in row])
        #plt.plot(amount_of_cooperate[-1])
value_parameter_0 = np.array(value_parameter_0)
plt.plot(sample_time_parameters, value_parameter_0, 'k--')
plt.xlabel('time step')
plt.ylabel('value of parameter 0')
plt.xlim([0, max_iteration])
plt.savefig(folder+'plots/value_parameter_0.pdf')
plt.close()


# -------------------------------------------------------

count_all_parameters = []
for i in range(98):
    origin_parameter_temp = []
    with open(folder+f"origin_parameter_{i}.csv") as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            origin_parameter_temp.append([float(i) for i in row])
            #plt.plot(amount_of_cooperate[-1])
    origin_parameter_temp = np.array(origin_parameter_temp)
    lenrow = origin_parameter_temp.shape[0]
    count_parameter_temp = np.zeros(lenrow)
    for i, row in enumerate(origin_parameter_temp):
        count_parameter_temp[i] = len(np.unique(row))
    #plt.plot(count_parameter_temp, 'k', alpha = 0.01)
    count_all_parameters.append(count_parameter_temp)
sum_count_all_parameters = np.sum(count_all_parameters,axis=0)
average_count_all_parameters = np.mean(count_all_parameters,axis=0)/150
std_count_all_parameters = np.std(count_all_parameters,axis=0)/150

fig = plt.figure(figsize=(4, 3))
plt.plot(sample_time_parameters, sum_count_all_parameters, 'k')
plt.xlabel('time step')
plt.ylabel('number of parameters in circulation')
plt.xlim([0, max_iteration])
plt.savefig(folder+'plots/parameters_in_circulation.pdf')
plt.close()

fig = plt.figure(figsize=(4, 3))
plt.plot(sample_time_parameters, average_count_all_parameters, 'k', label='average')
plt.fill_between(sample_time_parameters, average_count_all_parameters-std_count_all_parameters, 
                    average_count_all_parameters+std_count_all_parameters,alpha=0.2, color='r', label='std')
plt.xlabel('time step')
#plt.ylabel('average number of parameters \nin circulation')
plt.ylabel('parameter diversity')
plt.xlim([0, max_iteration])
plt.xticks([0,max_iteration//2,max_iteration])
plt.legend()
plt.savefig(folder+'plots/average_parameters_in_circulation.pdf')
plt.close()

fig = plt.figure(figsize=(4, 3))
for row in count_all_parameters:
    plt.plot(sample_time_parameters, row/150, alpha=0.5)
plt.ylabel('number of parameters \nin circulation')
plt.ylabel('parameter diversity')
plt.xlabel('time step')
plt.xlim([0, max_iteration])
plt.xticks([0,max_iteration//2,max_iteration])
plt.title('diversity of values per parameter')

#plt.legend()
plt.savefig(folder+'plots/all_parameters_in_circulation.pdf')
plt.close()

# -------------------------------------------------------

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



plt.figure()
plt.xlim([np.amin(position_x)-0.1,np.amax(position_x)+0.1])
plt.ylim([np.amin(position_y)-0.1,np.amax(position_y)+0.1])
plt.xticks([])
plt.yticks([])
#plt.xticks([0,1400])
#plt.yticks([0,800])
#plt.xlabel('x')
#plt.ylabel('y')

for i in range(n_agents):
    plt.plot(position_x[0:400,i], position_y[0:400,i], '-')
plt.savefig(folder+'plots/start_trajectories.pdf')
plt.close()



plt.figure()
plt.xlim([np.amin(position_x)-0.1,np.amax(position_x)+0.1])
plt.ylim([np.amin(position_y)-0.1,np.amax(position_y)+0.1])
plt.xticks([])
plt.yticks([])
#plt.xlabel('x')
#plt.ylabel('y')
for i in range(n_agents):
    plt.plot(position_x[-400:,i], position_y[-400:,i], '-')
plt.savefig(folder+'plots/final_trajectories.pdf')
plt.close()




# -------------------------------------------------------
"""
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
"""
# -------------------------------------------------------

idx_parameters = [0, 24, 53, 74]
fig = plt.figure(figsize=(4, 3))
value_parameter = {}
for idx in idx_parameters:
    value_parameter[idx] = []
    with open(folder+f"value_parameters_{idx}.csv") as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            value_parameter[idx].append([float(i) for i in row])
            #plt.plot(amount_of_cooperate[-1])
    value_parameter[idx] = np.array(value_parameter[idx])

# -------------------------------------------------------
def rangelabel(bins):
    #print(bins)
    lenbin = round(100*(bins[1]-bins[0]))/100
    #print(lenbin)
    strbin = [f'[{round(bin_start,2)}:{round(bin_start+lenbin, 2)}]' for bin_start in bins]
    return strbin

def survex(results, category_names, idx):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    #print(labels)
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(4, 3))
    #ax.invert_xaxis()
    #ax.yaxis.set_visible(False)
    #category_range_label = rangelabel(category_names)

    ax.set_ylim(0, np.sum(data, axis=1).max())
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        height = data[:, i]
        starts = data_cum[:, i] - height
        rects = ax.bar(labels, height, bottom=starts, width=100,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #ax.bar_label(rects, label_type='center', color=text_color)
    leg = ax.legend(ncol=2, fontsize='small', title=f'range of values', bbox_to_anchor=(1,1), loc='upper left') # for\nparameter {idx}
    leg._legend_box.align = "left"

    #ncol=len(category_names),
    ax.set_ylabel('distribution of parameter values')
    ax.set_xlabel('time step')
    ax.set_xlim([0,max_iteration])
    ax.set_xticks([0,max_iteration//2,max_iteration])
    title = ax.set_title(f'distribution of values for parameter {idx}', x=0.9)
    #title._title_box.align = "center"




    return fig, ax

#print(idx_parameters, value_parameter)
fig_subplot, axs = plt.subplots(2, 2)
idx_parameters = [24, 53, 74]
pos = [[0,0],[0,1],[1,0]]
for k,idx in enumerate(idx_parameters):
    hist_dict = {}
    for i in range(0,len(value_parameter[idx])):
        hist, bins = np.histogram(value_parameter[idx][i], 20, [-1,1])
        hist_dict[sample_time_parameters[i]] = hist/150

    fig, ax = survex(hist_dict, rangelabel(bins), idx)
    axs[pos[k][0],pos[k][1]] = ax
    plt.tight_layout()
    plt.savefig(folder+f'plots/value_param_{idx}_distribution.pdf')
    plt.close()
figlegend = plt.figure()

figlegend.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], ncol=3, fontsize='small', title=f'range of values')
axs[1,1] = figlegend
figlegend.savefig(folder+f'plots/legend.pdf')
fig_subplot.savefig(folder+f'plots/test.pdf')
    

