import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir

folder = "logs/2021-10-22-15-12-46-826143/"
fig = plt.figure()
exp_list = [s for s in listdir(folder) if '.pdf' not in s]

max_step = 14000
sample_time = []
with open( folder+exp_list[0]+"/sample_time.csv") as read_obj:
    csv_reader = list(csv.reader(read_obj))[0][:-1]
    sample_time = [int(i) for i in csv_reader]
score_all = []

for exp in exp_list:
    exp_folder = folder+exp+'/'
    score = []

    with open(exp_folder+"score.csv") as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            score.append(np.mean([float(i) for i in row]))
    if np.mean(score) < 1:
        print(exp)
    else:
        pass
    score_all.append(score)


score_all = np.array(score_all)
score_all_average = np.mean(score_all, axis=0)
score_all_std = np.std(score_all, axis=0)
median = np.median(score_all, axis=0)
quantile1 = np.quantile(score_all, 0.25, axis=0)
quantile3 = np.quantile(score_all, 0.75, axis=0)

plt.figure(figsize=(4,3))
#for i in range(len(score_all)):
#    plt.plot(sample_time, score_all[i,:], 'k')
#plt.plot(sample_time, score_all_average, 'r', label='average')
#plt.fill_between(sample_time, score_all_average-score_all_std, 
#                    score_all_average+score_all_std,alpha=0.2, color='r', label='standard deviation')

plt.fill_between(sample_time, quantile1, 
                    quantile3,alpha=0.2, color='k', label='inter-quartile range')
plt.plot(sample_time, median, 'k', label='median')

#plt.plot(sample_time, score)
plt.xlabel('time step')
plt.ylabel('average reward')
plt.xlim([0, max_step])
plt.ylim([0,4])
plt.xticks([0, max_step//2, max_step])
plt.legend(loc='upper left')
plt.title('swarm performance wrt foraging')
plt.savefig(folder+'score_median.pdf')
plt.close()


plt.figure(figsize=(4,3))
for i in range(len(score_all)):
    plt.plot(sample_time, score_all[i,:], 'k')

plt.xlabel('time step')
plt.ylabel('average reward')
plt.xlim([0, max_step])
plt.ylim([0,4])
plt.xticks([0, max_step//2, max_step])
plt.title('swarm performance wrt foraging')
plt.savefig(folder+'score_alltraces.pdf')
plt.close()