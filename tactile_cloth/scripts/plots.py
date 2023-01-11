import sys
import argparse
import numpy as np 
import matplotlib.pyplot as plt 

def plot_magnetometer(data, fname):
    fig, axes = plt.subplots(nrows = 6, ncols = 1, sharex = True)
    names = {0:"Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    axes[-1].set_title('Automatic Label')
    axes[-1].plot(data[:,-2], 'black', label="Contact Data")
    for i in range(5):
        axes[i].plot(data[:,i*3], 'r', label="Bx")
        axes[i].plot(data[:,i*3+1], 'g', label = "By")
        axes[i].plot(data[:,i*3+2], 'b', label = "Bz")
        axes[i].set_title(names[i])
        pass
    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    fig.set_size_inches(20, 11.3)  
    plt.savefig(fname,  bbox_inches='tight', dpi = 100)
    plt.close()

def plot_magnetometer_by_row(data, fname):
    """
    data: n x 17 (1-15 data, 16 str, 17 label / prediction)
    """
    fig, axes = plt.subplots(nrows = 5, ncols = 3)
    
    names = {0:"Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    magns = np.array([ # 5 x n x 3 (xyz)
        data[:, 0:3],
        data[:, 3:6],
        data[:, 6:9],
        data[:, 9:12],
        data[:, 12:15],
    ])

    # plot each magnetometer
    for i in range(5):
        axes[i][0].set_title(names[i] + " X")
        axes[i][0].plot(magns[i, :, 0])
        axes[i][1].set_title(names[i] + " Y")
        axes[i][1].plot(magns[i, :, 1])
        axes[i][2].set_title(names[i] + " Z")
        axes[i][2].plot(magns[i, :, 2])

    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    # fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    # fig.set_size_inches(20, 11.3)  
    plt.savefig(fname,  bbox_inches='tight', dpi = 100)
    plt.close()

def juxtaposition_plot(data1, data2, figname):
    fig, axes = plt.subplots(nrows = 6, ncols = 2, sharex = True)
    data = [data1, data2]
    for i in range(2):
        for j in range(5):
            axes[j,i].plot(data[i][:,j*3], 'r', label="Bx")
            axes[j,i].plot(data[i][:,j*3+1], 'g', label = "By")
            axes[j,i].plot(data[i][:,j*3+2], 'b', label = "Bz")
        pass
    axes[-1,0].plot(data1[:,-2], 'black', label="Contact Data")
    axes[-1,1].plot(data2[:,-2], 'black', label="Contact Data")
    lines, labels = axes[-2,-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =16.0)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 16.0)
    fig.suptitle(figname, fontsize = 16)
    plt.show()
    pass

def juxtaposition_plot_with_indices(data1, data2, figname, indices):
    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharey = True)
    data = [data1, data2]
    for i in range(2):
        print(indices)
        axes[i].plot(data[i][:,int(indices[0])], 'r', label="By Left")
        axes[i].plot(data[i][:,int(indices[1])], 'g', label="Bx Right")
        axes[i].plot(data[i][:,int(indices[2])], 'b', label="Bx Left")
        axes[i].plot(data[i][:,int(indices[3])], 'y', label="By Center")

        pass
    lines, labels = axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =16.0)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 16.0)
    fig.suptitle(figname, fontsize = 16)
    plt.show()
    pass

def get_data(name, trial):
    classification_folder = "/home/sashank/catkin_ws/src/tactilecloth/classification_data/"
    dpath = classification_folder+name+"/"+str(trial)+"/"+str(trial)+"_reskin_data.csv"
    data_arr = np.loadtxt(dpath, delimiter=",")
    return data_arr

def get_trial_data(path):
    return np.loadtxt("%s/reskin_data.csv" % path, delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="path to trial", 
        type=str, default=None)
    args = parser.parse_args()

    trial_data = get_trial_data(args.path)
    plot_magnetometer(data=trial_data, fname=args.path + "/magnetometers.png")
    plot_magnetometer_by_row(data=trial_data, fname=args.path + "/magnetometers_by_component.png")
    
    sys.exit(1)


    # data = get_data("0cloth_21Jan", 1)
    # plot_magnetometer(data)
    # classification_folder = "/home/sashank/catkin_ws/src/tactilecloth/classification_data/"
    # names = ["0cloth_21Jan", "1cloth_21Jan"]
    # for name in names:
    #     for i in range(15):
    #         data = get_data(name, i+1)
    #         fname = classification_folder+name+"/"+str(i+1)+"/all_magnetometers_plot.png"
    #         plot_magnetometer(data, fname)
    classification_folder = "/home/sashank/catkin_ws/src/tactilecloth/classification_data/"
    i= 5
    j = 5
    f1 = "1clothrub"
    f2 = "2clothrub"
    d1 = np.loadtxt(classification_folder+f1+"/"+f1+"_reskin_data.csv", delimiter=",")
    d2 = np.loadtxt(classification_folder+f2+"/"+f2+"_reskin_data.csv", delimiter=",")
    figname = f1+" vs "+f2
    # juxtaposition_plot(d1, d2, figname)
    indices = [13,6,12,1]
    juxtaposition_plot(d1, d2, figname)

    # i= 14
    # j = 14
    # f1 = "1cloth_7feb"
    # f2 = "2cloth_7feb"
    # d1 = get_data(f1, i)
    # d2 = get_data(f2, j)
    # figname = f1+" Trial: "+str(i) + " vs "+f2+" Trial: "+str(j)
    # # juxtaposition_plot(d1, d2, figname)
    # indices = [13,6,12,1]
    # juxtaposition_plot(d1, d2, figname)

    # i= 3
    # j = 3
    # f1 = "1cloth_7feb"
    # f2 = "2cloth_7feb"
    # d1 = get_data(f1, i)
    # d2 = get_data(f2, j)
    # figname = f1+" Trial: "+str(i) + " vs "+f2+" Trial: "+str(j)
    # # juxtaposition_plot(d1, d2, figname)
    # indices = [13,6,12,1]
    # juxtaposition_plot(d1, d2, figname)
    
    
    

