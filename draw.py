import matplotlib.pyplot as plt
import numpy as np
import math



def draw_algo_list_log_x_axis(x_list,Algo_mean_list,Algo_std_list,Algo_label,fname='ratio.jpg',x_label='{}'.format(chr(955)),y_label = 'Ratio',ylim = None,line_style=None):
    plt.cla()
    fname = 'pic/{}'.format(fname)
    color_list = ['b','r', 'y', 'g','m',  'c','#FF00FF','#CEFFCE','#D2691E']
    marker_list = ['o', 'v', '^', '<', '>', 's','3','8','|','x']
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if ylim != None:
        plt.ylim(ylim[0],ylim[1])
    x_list = [x+1 for x in range(len(x_list))]
    fake_x_list = [r'$2^{'+str(x-1)+'}$' for x in x_list]

    plt.plot(x_list, [1.0 for _ in x_list], color='k',linestyle=(0, (5, 10)), linewidth=1)

    for i in range(len(Algo_label)):
        plt.errorbar(x_list, Algo_mean_list[i], yerr=np.array(Algo_std_list[i])*0.1, ecolor=color_list[i],fmt='none')
        if line_style == None:
            if i %2 == 1:
                ls = '-'
            else:
                ls = '--'
        else:
            ls = line_style[i]
        if Algo_label[i] in ['G']:
            #pass
            plt.plot(x_list, Algo_mean_list[i], color=color_list[i],linestyle=ls, linewidth=1, label=Algo_label[i])
        else:
            #pass
            plt.plot(x_list, Algo_mean_list[i], color=color_list[i],linestyle=ls, linewidth=1,label = Algo_label[i]) # marker=marker_list[i]

    plt.xticks(x_list,fake_x_list)
    if len(Algo_label) > 1:
        plt.legend(loc='upper right')
    plt.savefig(fname, dpi=200)

def draw_algo_list(x_list,Algo_mean_list,Algo_std_list,Algo_label,fname='ratio.jpg',x_label='{}'.format(chr(955)),y_label = 'Ratio',ylim = None,line_style=None):
    plt.cla()
    fname = 'pic/{}'.format(fname)
    color_list = ['b','r', 'y', 'g','m',  'c','#FF00FF','#CEFFCE','#D2691E']
    marker_list = ['o', 'v', '^', '<', '>', 's','3','8','|','x']
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if ylim != None:
        plt.ylim(ylim[0],ylim[1])


    plt.plot(x_list, [1.0 for _ in x_list], color='k',linestyle=(0, (5, 10)), linewidth=1)

    for i in range(len(Algo_label)):
        plt.errorbar(x_list, Algo_mean_list[i], yerr=np.array(Algo_std_list[i])*0.1, ecolor=color_list[i],fmt='none')
        if line_style == None:
            if i %2 == 1:
                ls = '-'
            else:
                ls = '--'
        else:
            ls = line_style[i]
        if Algo_label[i] in ['G']:
            #pass
            plt.plot(x_list, Algo_mean_list[i], color=color_list[i],linestyle=ls, linewidth=1, label=Algo_label[i])
        else:
            #pass
            plt.plot(x_list, Algo_mean_list[i], color=color_list[i],linestyle=ls, linewidth=1,label = Algo_label[i]) # marker=marker_list[i]


    if len(Algo_label) > 1:
        plt.legend(loc='upper right')
    plt.savefig(fname, dpi=200)


def draw_single_algo(x_list,Algo_mean_list,Algo_std_list,fname='ratio.jpg',x_label='{}'.format(chr(955)),y_label = 'Ratio',ylim = None):
    plt.cla()
    fname = 'pic/{}'.format(fname)
    color_list = ['k','r', 'm', 'y', 'g', 'b', 'c','#FF00FF','#CEFFCE','#D2691E']
    marker_list = ['o', 'v', '^', '<', '>', 's','3','8','|','x']
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if ylim != None:
        plt.ylim(ylim[0],ylim[1])
    plt.errorbar(x_list, Algo_mean_list, yerr=np.array(Algo_std_list)*0.1, ecolor=color_list[0],fmt='none')
    plt.plot(x_list, Algo_mean_list, color=color_list[0],linestyle='-', linewidth=1)


    plt.savefig(fname, dpi=200)

































