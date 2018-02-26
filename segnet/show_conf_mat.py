
import os
import time
import numpy as np
import matplotlib

#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle

label_names = ["Sky", "Building", "Pole", "Road marking", "Road", "Pavement", "Tree", "SignSymbol", "Fence",
               "Car", "Pedestrian", "Bicyclist", "Unlabelled"]


def show_confusion_matrix(conf_arr, name):

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

        
    fig = plt.figure(1, figsize=(10,10))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_title("Confusion matrix for Epoch: " + name)    
    
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.get_cmap("hot"), 
                    interpolation='nearest')


    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(norm_conf[y][x]*1000)), xy=(x,y),
                            horizontalalignment='center',
                            verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = label_names
    
#    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])

    plt.show(block=False)


def load_confusion_matrix(file):
    with open(file, "rb") as pf:
        matrix = pickle.load(pf)
        pf.close()
        return matrix

if __name__ == '__main__':


    plt.ion()
    for file in sorted(os.listdir("results/")):
        print(file)

        cm = load_confusion_matrix("results/"+file)
        show_confusion_matrix(cm, file)
        key = input("Q to break")
        if key == "Q":
            break       

        
