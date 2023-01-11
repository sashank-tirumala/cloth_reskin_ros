"""
Put utility files here to support `reskin_classify.py`.
Basing this off of Sashank's script:
    Feb11_Data_Analysis-checkpoint.ipynb

(c) Daniel Seita
"""
import os
from os.path import join
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
from collections import defaultdict
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T


def print_class_counts(y, data_type):
    y_unique = np.sort(np.unique(y))
    print('Unique class labels for y: {} in {}'.format(y_unique, data_type))
    for k in y_unique:
        print('  count {}: {}'.format(int(k), np.sum(y==k)))
    print()


def get_data(dir_name):
    """Given a directory (a single 'episode') get the csv data.

    Returns numpy array of CSV data, of shape (N,15) where N is the time, which is
    usually around 2000. See `plot_magnetometer` for interpreting the data.
    """
    file_name = join(dir_name, 'reskin_data.csv')
    data = np.loadtxt(file_name, delimiter=",")
    data = data[:,:-1]
    return data


def get_data_XY(dir_name):
    """Same as `get_data` but now returning Sashank's labels."""
    file_name = join(dir_name, 'reskin_data_estim_gt.csv')
    data = np.loadtxt(file_name, delimiter=",")
    X = data[:,:-1]
    Y = np.reshape(data[:,-1],(-1,1))
    return X,Y


def get_data_XY_new(dir_name, get_images=False):
    """Daniel: he must add a 17th index here, guess like data changed?

    Extending this to support video data. We use video capture to read the .avi
    files, which will give us frames:
        frame = frame[:480, :, :]  # for an image-based classifier.
        frame = frame[480:, :, :]  # magnetometers (keep as future reference).
    The frame count will be lower than the ReSkin counts, so we sub-sample.

    Big assumption: that the frames are spread out 'uniformly' throughout the
    training. Will need to confirm with Sashank about this.

    For quick and dirty testing:
        import moviepy
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(list(frames), fps=20)
        clip.write_gif('test.gif', fps=20)

    Careful! For some datasets, they used `reskin_labelled.csv`.
    Though I hope this is just for 1 dataset.
    """
    if 'franka_norub_folded_random_18feb' in dir_name:
        file_name = join(dir_name, 'reskin_labelled.csv')  # for feb18 data
    else:
        file_name = join(dir_name, 'reskin_data.csv')  # other data
    data = np.loadtxt(file_name, delimiter = ",")
    X = data[:,:-2]
    Y = np.reshape(data[:,-2],(-1,1))

    if get_images:
        video_name = join(dir_name, 'videos/magnetometers.avi')
        vcap = cv2.VideoCapture(video_name)
        ret = True
        frames = []
        while ret:
            ret, frame = vcap.read()
            if frame is not None:
                frames.append(frame)
        X = np.array(frames)  # (n_frames, 960, 640, 3)
        subsamp = np.linspace(start=0, stop=len(Y)-1, num=len(frames))
        subsamp = subsamp.astype(np.int32)
        Y = Y[subsamp, :]   # Y.shape --> (n_frames, 1)

    return X,Y


def plot_magnetometer(data):
    """Shows all magnetometer readings as a function of time (x-axis)."""
    fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
    names = {0: "Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    for i in range(5):
        axes[i].plot(data[:,i*3],   'r', label="Bx")
        axes[i].plot(data[:,i*3+1], 'g', label="By")
        axes[i].plot(data[:,i*3+2], 'b', label="Bz")
        axes[i].set_title(names[i])
    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    fig.set_size_inches(20, 11.3)
    plt.show()


def plot_diff(data):
    """Haven't tested, but likely for identifying points of contact / no contact.

    For points of contact, the `plot_norm` might be suitable.
    """
    fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
    names = {0: "Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    dx = 0.001
    for i in range(5):
        axes[i].plot(np.diff(data[:,i*3])/dx,   'r', label="Bx")
        axes[i].plot(np.diff(data[:,i*3+1])/dx, 'g', label="By")
        axes[i].plot(np.diff(data[:,i*3+2])/dx, 'b', label="Bz")
        axes[i].set_title(names[i])
    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    fig.set_size_inches(20, 11.3)
    plt.show()


def plot_norm(data):
    """Squares the data, then plots _differences_ across consecutive time steps.

    I think this is used to determine points of contact / no contacts? I think if there
    is a lot of contact, the norm would be higher.
    """
    fig = plt.figure()
    sqdata = np.square(data)
    norm_data = np.diff(np.mean(sqdata, axis = 1))
    plt.plot(norm_data, 'r', label="norm")
    fig.set_size_inches(20, 11.3)
    plt.show()


def label_data(data):
    """From Sashank, I think he's using this for labeling data."""
    sqdata = np.square(data)
    diff_norm_data = np.diff(np.mean(sqdata, axis = 1))


def get_cloth_dir_names(p0=None, p1=None, p2=None):
    """From Sashank. Extending to support optional directories."""
    cloth0_dir_names = []
    cloth1_dir_names = []
    cloth2_dir_names = []
    if p0 is not None:
        for dir in sorted(os.listdir(p0)):
            cloth0_dir_names.append(p0+"/"+dir)
    if p1 is not None:
        for dir in sorted(os.listdir(p1)):
            cloth1_dir_names.append(p1+"/"+dir)
    if p2 is not None:
        for dir in sorted(os.listdir(p2)):
            cloth2_dir_names.append(p2+"/"+dir)
    return cloth0_dir_names, cloth1_dir_names, cloth2_dir_names


# deprecated?
def create_cloth_datas(cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, max_x_len, new_collect = True):
    """From Sashank. (I think we use other methods, though)"""
    cloth0_datas = []
    cloth1_datas = []
    cloth2_datas = []

    for dir_name in cloth0_dir_names:
        if(new_collect):
            x,y = get_data_XY_new(dir_name)
        else:
            x,y = get_data_XY(dir_name)
        x = pad_time_series(x, max_x_len)
        cloth0_datas.append(x)

    for dir_name in cloth1_dir_names:
        if(new_collect):
            x,y = get_data_XY_new(dir_name)
        else:
            x,y = get_data_XY(dir_name)
        x = pad_time_series(x, max_x_len)
        cloth1_datas.append(x)

    for dir_name in cloth2_dir_names:
        if(new_collect):
            x,y = get_data_XY_new(dir_name)
        else:
            x,y = get_data_XY(dir_name)
        x = pad_time_series(x, max_x_len)
        cloth2_datas.append(x)
    return cloth0_datas, cloth1_datas, cloth2_datas


# deprecated?
def get_dataset_dir_names(cloth0, cloth1, cloth2, num_train):
    """From Sashank, I modified only so that we might shuffle directories."""
    train_dir_names = cloth0[0:num_train] + \
                      cloth1[0:num_train] + \
                      cloth2[0:num_train]
    test_dir_names = cloth0[num_train:] + \
                     cloth1[num_train:] + \
                     cloth2[num_train:]
    return train_dir_names, test_dir_names


def get_raw_data(train_dir_names, test_dir_names, new_collect=True, get_images=False):
    """From Sashank, forms the raw (X,Y,x,y) data (train upper, test lower).

    Extending this to support a `get_images` method for video data, by supplying it
    as an argument to `get_data_XY_new()`. We still use the same 'vstack'-ing, etc.

    Also supporting test_drift so we can record the indices of (test) episodes. We
    can just do that by default, it's cheap to record indices.
    """
    X = []
    Y = []
    for dir_name in train_dir_names:
        # Each of these episodes has 2 labels, (a) grasping nothing, and (b) the
        # label of {0,1,2} cloth, where 0 here means ReSkin touches the other tip.
        if new_collect:
            x,y = get_data_XY_new(dir_name, get_images=get_images)
        else:
            x,y = get_data_XY(dir_name)
        X.append(x)
        Y.append(y)
    if len(X) > 0:
        X = np.vstack(X)
        Y = np.vstack(Y)

    x_test = []
    y_test = []
    info = []
    for dir_name in test_dir_names:
        if new_collect:
            x,y = get_data_XY_new(dir_name, get_images=get_images)
        else:
            x,y = get_data_XY(dir_name)
        x_test.append(x)
        y_test.append(y)
        data_info = (len(y), dir_name)
        info.append(data_info)
    if len(x_test) > 0:
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
    return X, Y, x_test, y_test, info


def normalize_data(X, x_test):
    """From Sashank. Fit only the train data (important!)."""
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    x_test = scaler.transform(x_test)
    return X, x_test


def get_dataset_from_dir_names(cloth0_dir_names, cloth1_dir_names, cloth2_dir_names,
        n_epis, n_train, new_collect=True, get_images=False, test_drift=0):
    """From Sashank, but making some modifications.

    Update: also adding the option to get image data, not sure how easily it will be.
    Args:
        test_drift: Only true if we do NOT want to shuffle.
    """
    assert len(cloth0_dir_names) == len(cloth1_dir_names) == len(cloth2_dir_names)

    # Normally we WANT to shuffle the episodic data (i.e., `test_drift = 0`).
    if test_drift == 0:
        ss_c0 = np.random.permutation(n_epis)
        ss_c1 = np.random.permutation(n_epis)
        ss_c2 = np.random.permutation(n_epis)
    else:
        ss_c0 = np.arange(n_epis)
        ss_c1 = np.arange(n_epis)
        ss_c2 = np.arange(n_epis)
    tr_c0, te_c0 = ss_c0[ :n_train ], ss_c0[ n_train: ]
    tr_c1, te_c1 = ss_c1[ :n_train ], ss_c1[ n_train: ]
    tr_c2, te_c2 = ss_c2[ :n_train ], ss_c2[ n_train: ]
    print('Train/Test epis (c0): {}, {}'.format(tr_c0, te_c0))
    print('Train/Test epis (c1): {}, {}'.format(tr_c1, te_c1))
    print('Train/Test epis (c2): {}, {}'.format(tr_c2, te_c2))
    cloth0_shuffled = []
    cloth1_shuffled = []
    cloth2_shuffled = []
    for idx in range(n_epis):
        cloth0_shuffled.append( cloth0_dir_names[ss_c0[idx]] )
        cloth1_shuffled.append( cloth1_dir_names[ss_c1[idx]] )
        cloth2_shuffled.append( cloth2_dir_names[ss_c2[idx]] )

    # Since we shuffled these, we can just take first n_train and use that for train.
    train_dir_names, test_dir_names = get_dataset_dir_names(
            cloth0_shuffled, cloth1_shuffled, cloth2_shuffled, n_train)
    print('len(train_dir_names): {}'.format(len(train_dir_names)))
    print('len(test_dir_names):  {}'.format(len(test_dir_names)))

    # Now we can get the raw data. DO NOT NORMALIZE, do that later!
    X, Y, x, y, info = get_raw_data(train_dir_names, test_dir_names, new_collect,
        get_images=get_images)

    # Return train, test, info.
    return (X, Y, x, y, info)


def pad_time_series(data, new_len):
    """From Sashank."""
    val = new_len - data.shape[0]
    if(val%2 == 0):
        left_pad_values = int(val/2)
        right_pad_values = int(val/2)
    elif(val%2 == 1):
        left_pad_values = int(val/2)
        right_pad_values = int(val/2)+1
    new_data = np.zeros((new_len, 15))
    for j in range(data.shape[1]):
        new_data[:,j] = np.pad(data[:,j], (left_pad_values, right_pad_values), mode = "edge")
    return new_data


# ---------------------------------------------------------------------- #
# For deep networks                                                      #
# ---------------------------------------------------------------------- #

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def debug_transf_images(data_imgs, is_valid=False):
    """Debug transformed images in a batch of data images.

    The idea is maybe we can use this to check if the transforms make sense, etc.
    To undo the normalization ... this seems to make sense.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # This undoes the normalization, should be in [0,1] now due to ToTensor().
    orig_imgs = unnormalize(data_imgs)
    orig_imgs = (orig_imgs.numpy() * 255).astype(np.uint8)  # (batch, 3, 224, 224)
    orig_imgs = np.transpose(orig_imgs, (0, 2, 3, 1))
    for o_img in orig_imgs:
        c = len([x for x in os.listdir('.') if 'transf_img_debug' in x])
        cv2.imwrite('transf_img_debug_{}_{}.png'.format(is_valid, c), o_img)

# ---------------------------------------------------------------------- #
# Deprecated methods                                                     #
# ---------------------------------------------------------------------- #

def _get_processed_reskin(reskin_c1, reskin_c2, n_epis, n_train, ignore_zero=False):
    """Process data into numpy. Might do for the PointNet++ as well.

    ignore_zero: ignore 0 cloth data, which we should use for PointNet++.
    """
    ss_c1 = np.random.permutation(n_epis)
    ss_c2 = np.random.permutation(n_epis)
    t_idxs_c1 = ss_c1[ :n_train ]
    t_idxs_c2 = ss_c2[ :n_train ]
    v_idxs_c1 = ss_c1[ n_train: ]
    v_idxs_c2 = ss_c2[ n_train: ]
    print('Train, valid epis (c1): {}, {}'.format(t_idxs_c1, v_idxs_c1))
    print('Train, valid epis (c2): {}, {}'.format(t_idxs_c2, v_idxs_c2))
    X_train, y_train = [], []
    X_valid, y_valid = [], []

    # reskin_c1: has labels 0 and 2, for no cloth and 1 cloth.
    for ridx, rdata in enumerate(reskin_c1):
        X,y = rdata  # X is (N,15), N is typically 1800-ish?
        assert np.sum(y==1) == 0, np.sum(y==1)
        assert np.sum(y==3) == 0, np.sum(y==3)
        if ignore_zero:
            X = X[np.where(y!=0)[0], :]
            y = y[y!=0][:,None]  # shoudl be (N,1)
        if ridx in t_idxs_c1:
            X_train.append(X)
            y_train.append(y)
        else:
            X_valid.append(X)
            y_valid.append(y)

    # reskin_c2: has labels 0 and 3, for no cloth and 2 cloth.
    for ridx, rdata in enumerate(reskin_c2):
        X,y = rdata  # X is (N,15), N is typically 1800-ish?
        assert np.sum(y==1) == 0, np.sum(y==1)
        assert np.sum(y==2) == 0, np.sum(y==2)
        if ignore_zero:
            X = X[np.where(y!=0)[0], :]
            y = y[y!=0][:,None]  # shoudl be (N,1)
        if ridx in t_idxs_c2:
            X_train.append(X)
            y_train.append(y)
        else:
            X_valid.append(X)
            y_valid.append(y)

    # Collect data. Shapes: (N,15) for X, (N,1) for y.
    X_train = np.vstack(X_train)
    X_valid = np.vstack(X_valid)
    y_train = np.vstack(y_train)
    y_valid = np.vstack(y_valid)
    data = {'X_train': X_train,
            'X_valid': X_valid,
            'y_train': y_train,
            'y_valid': y_valid}
    return data


def classify_reskin(reskin_c1, reskin_c2, n_folds=50, debug_print=True, ignore_zero=True):
    """Trying to reproduce Sashank's results, and also maybe try other methods.

    From my past history, I like Random Forests. :-) My first Berkeley paper!
    We have a for loop over folds where each time we randomize which episodes get
    assigned to train vs valid.
    """
    METHOD = 'kNN'
    assert METHOD in ['RF', 'kNN']
    assert len(reskin_c1) == len(reskin_c2), len(reskin_c1)
    n_epis = len(reskin_c1)
    n_train = int(n_epis * 0.75)
    print('About to classify ReSkin, n_train / n_epis: {} / {}'.format(n_train, n_epis))
    print('Method: {}'.format(METHOD))
    stats = defaultdict(list)
    if ignore_zero:
        stats['cmat'] = np.zeros((2,2))
    else:
        stats['cmat'] = np.zeros((3,3))

    # If deleting features? Which ones did Sashank say did not matter? Edit:  Bz, but
    # those were with older grippers. And with only Bz we still get really good results!
    delete_features = False  # if True, uncomment desired choice below.
    idxs_keep = np.arange(15)  # all
    #idxs_keep = np.array( [1,2,4,5,7,8,10,11,13,14] )  # ignore all Bx (0,3,6,9,12)
    #idxs_keep = np.array( [0,2,3,5,6,8,9,11,12,14] )  # ignore all By (1,4,7,10,13)
    #idxs_keep = np.array( [0,1,3,4,6,7,9,10,12,13] )  # ignore all Bz (2,5,8,11,14)
    #idxs_keep = np.array( [0,3,6,9,12] )  # ignore all By AND Bz (just Bx)
    #idxs_keep = np.array( [1,4,7,10,13] )  # ignore all Bx AND Bz (just By)
    #idxs_keep = np.array( [2,5,8,11,14] )  # ignore all Bx AND By (just Bz)

    for i in range(n_folds):
        print('\n|------------------------- Evaluation {} --------------------------|'.format(i+1))
        # New data for this run, randomize train / valid for classes 1 and 2.
        # Probably not necessary to randomize both c1 and c2, but doesn't hurt?
        data_np = _get_processed_reskin(reskin_c1, reskin_c2, n_epis, n_train,
            ignore_zero=ignore_zero)
        X_train = data_np['X_train']
        X_valid = data_np['X_valid']
        y_train = data_np['y_train']
        y_valid = data_np['y_valid']
        if delete_features:
            X_train = X_train[:, idxs_keep]
            X_valid = X_valid[:, idxs_keep]
        if debug_print:
            print('X_train: {}'.format(X_train.shape))
            print('X_valid: {}'.format(X_valid.shape))
            print('y_train: {}'.format(y_train.shape))
            print('y_valid: {}'.format(y_valid.shape))
            print('Total train + valid: {}'.format(len(y_train)+len(y_valid)))

        # Transforms (N,15) X_train data so mean + std are 0 and 1. X_valid is
        # also transformed, but follows X_train stats (not perfectly normalized).
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)

        if METHOD == 'RF':
            clf = RandomForestClassifier(max_depth=None, random_state=0)
        elif METHOD == 'kNN':
            clf = neighbors.KNeighborsClassifier(10, weights="distance")

        # Scipy / sklearn uses a consistent API, fortunately.
        clf.fit(X_train, y_train.ravel())  # Fit according to the training data.
        y_vpred = clf.predict(X_valid)
        score = balanced_accuracy_score(y_valid, y_vpred)
        best_cf = confusion_matrix(y_valid, y_vpred)
        print('Balanced accuracy score: {}'.format(score))
        print(best_cf)
        stats['score'].append(score)
        stats['cmat'] += best_cf  # keep ADDING so we can divide

    print('\n-------------------------------------------------------')
    print('Done over {} different folds.'.format(n_folds))
    print('Score: {} +/- {}'.format(
            np.mean(stats['score']), np.std(stats['score'])))
    print('Avg confusion matrix:\n{}'.format(stats['cmat'] / n_folds))
