"""
ReSkin classifier, including image-based classifier.
Basing this off of Sashank's script:
    Feb11_Data_Analysis-checkpoint.ipynb

(c) Daniel Seita
"""
import cv2
import os
from os.path import join
import shutil
import argparse
import pickle
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torchvision import datasets, transforms as T
import torchvision.models as models
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4, linewidth=120, edgeitems=100)
# from scripts import reskin_utils as U
import reskin_utils as U

# ----------------------------------------------------------------------- #
import matplotlib.pyplot as plt
FIGSIZE = (6, 6)
DPI = 100
xsize = 20
ysize = 20
xticksize = 14
yticksize = 14
titlesize = 22
legendsize = 16
lw = 3
# ----------------------------------------------------------------------- #

# ----------------------------------------------------------------------- #
# Data directories. CHANGE AS NEEDED! Each subdir has a `videos/magnetometers.avi`
# file that we can use for extracting image-based data.
# HEAD =  '/data/seita/softgym_ft'
HEAD =  '/media/tweng/ExtraDrive2/fabric_touch/bagfiles'

##RESKIN_EXP = 'franka_norub_folded_random'
##CLASSES = {
##    0: join(HEAD, RESKIN_EXP, '0cloth_norub_auto'),
##    1: join(HEAD, RESKIN_EXP, '1cloth_norub_auto'),
##    2: join(HEAD, RESKIN_EXP, '2cloth_norub_auto'),
##}

##RESKIN_EXP = 'franka_norub_folded_random_18feb'
##CLASSES = {
##    0: join(HEAD, RESKIN_EXP, '0cloth_norub_auto_0'),
##    1: join(HEAD, RESKIN_EXP, '1cloth_norub_auto_0'),
##    2: join(HEAD, RESKIN_EXP, '2cloth_norub_auto_0'),
##}
##TESTDRIFT = join(HEAD, '0cloth_norub_auto_robot_0')

RESKIN_EXP = 'franka_folded_random_autorobot'
CLASSES = {
    0: join(HEAD, RESKIN_EXP, '0cloth_norub_auto_robot_0'),
    1: join(HEAD, RESKIN_EXP, '1cloth_norub_auto_robot_0'),
    2: join(HEAD, RESKIN_EXP, '2cloth_norub_auto_robot_0'),
}

for key in sorted(CLASSES.keys()):
    print(f'Cloth dirs {key}: {CLASSES[key]}')
# ----------------------------------------------------------------------- #


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def classify_reskin_img(cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, args):
    """In a similar format as `classify_reskin` but using image-classifiers now.

    References for PyTorch / ResNet:
        https://pytorch.org/vision/stable/models.html
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        https://discuss.pytorch.org/t/questions-about-imagefolder/774/2
        https://pytorch.org/tutorials/beginner/saving_loading_models.html

    Note: as of 02/17 data collection, we should NOT have to crop, as Sashank and Thomas
    adjusted the camera angle so that it's zoomed in at the right spot.
    """
    data_dir = 'tmp'
    feature_extract = False
    num_classes = 4

    # Get image-based data set up for neural network. Careful not to override these values!
    X_data, Y_train, x_data, Y_valid, _ = U.get_dataset_from_dir_names(cloth0_dir_names,
        cloth1_dir_names, cloth2_dir_names, args.n_epis, args.n_train, get_images=True)

    # NOTE(daniel): the X's here actually are stacked (img on top, matplotlib on bottom).
    # Be careful not to override `x` until AFTER making the ImageFolder. Also the labels {Y,y}.
    # The images have height 960, so we take only the top half.
    X_train, _ = X_data[:, :480, :, :], X_data[:, 480:, :, :]  # (images, matplotlib) upper
    X_valid, _ = x_data[:, :480, :, :], x_data[:, 480:, :, :]  # (images, matplotlib) lower

    # Cropping will have to be tuned a bit for each dataset :-(.
    # Make sure train/valid have the same cropping! We can inspect this with ImageFolder.
    if 'franka_norub_folded_random_18feb' in cloth0_dir_names[0]:
        # This one needs some uniform cropping.
        X_train = X_train[:, 20:460, 20:620, :]
        X_valid = X_valid[:, 20:460, 20:620, :]
    elif 'franka_folded_random_autorobot' in cloth0_dir_names[0]:
        # Needs heavy cropping of the _left_ portion, and some of the top.
        X_train = X_train[:, 80:, 240:, :]  # makes 400x400 img
        X_valid = X_valid[:, 80:, 240:, :]  # makes 400x400 img
    else:
        raise NotImplementedError()
    print(f'\nTrain (after cropping): {X_train.shape}, {Y_train.shape}')
    print(f'Valid (after cropping): {X_valid.shape}, {Y_valid.shape}')
    print(f'Total train/test: {len(Y_train)+len(Y_valid)}')
    U.print_class_counts(Y_train, data_type='train')
    U.print_class_counts(Y_valid, data_type='test')

    import IPython; IPython.embed()

    # --------------------------------------------------------------------------------- #
    # Use the ImageFolder data format for PyTorch, also helps us inspect the labels.
    print("\nCreating data for ImageFolder...")
    if os.path.exists(data_dir):
        print(f'Deleting: {data_dir} ...')
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    os.mkdir(join(data_dir, 'train'))
    os.mkdir(join(data_dir, 'valid'))
    for c in range(num_classes):
        os.mkdir(join(data_dir, f'train/{c}'))
        os.mkdir(join(data_dir, f'valid/{c}'))

    # Train ImageFolder, may take some time.
    for ii in range(len(Y_train)):
        X_item = X_train[ii]
        Y_item = int(Y_train[ii])
        save_dir = join(data_dir, f'train/{Y_item}')
        count = len([z for z in os.listdir(save_dir) if '.png' in z])
        fname = join(save_dir, f'{str(count).zfill(4)}.png')
        cv2.imwrite(fname, X_item)

    # Valid ImageFolder, may take some time.
    print('Done with training, now validation...')
    for ii in range(len(Y_valid)):
        x_item = X_valid[ii]
        y_item = int(Y_valid[ii])
        save_dir = join(data_dir, f'valid/{y_item}')
        count = len([z for z in os.listdir(save_dir) if '.png' in z])
        fname = join(save_dir, f'{str(count).zfill(4)}.png')
        cv2.imwrite(fname, x_item)
    # --------------------------------------------------------------------------------- #

    # Load the PRETRAINED ResNet18 model.
    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    model_ft = model_ft.to(args.device)
    print(f'Parameter count to train: {U.count_parameters(model_ft)}')

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. with requires_grad True.
    params_to_update = model_ft.parameters()
    print("\nParams to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)

    # Data augmentation and normalization for training (just norm for valid).
    # Note: validation will resize and center crop, but it resizes to the same
    # thing, hence there's no cropping involved. But training will crop, so that
    # means training might be a bit different. Though the discrepancy can be cut
    # down a bit if we end up cropping parts of the original images.
    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets, these need the transforms!
    print(f"Initializing Datasets and Dataloaders in {data_dir} ...")
    train_data = datasets.ImageFolder(join(data_dir,'train'), data_transforms['train'])
    valid_data = datasets.ImageFolder(join(data_dir,'valid'), data_transforms['valid'])
    print(f'datasets.ImageFolder len(train_data): {len(train_data)}')
    print(f'datasets.ImageFolder len(valid_data): {len(valid_data)}')

    # Create training and validation dataloaders using these datasets. Do NOT shuffle
    # the validation data, we want that off so we can accurately compare later.
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # --------------------- evaluation before training ------------------------ #
    # Do this as a sanity check to get poor values (random guessing).
    print('\nEvaluating, then will train!')
    model_ft.eval()
    correct_v = 0
    valid_gt = []
    for inputs, labels in valid_loader:
        #U.debug_transf_images(inputs, is_valid=True)  # useful for debugging
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
        correct_v += torch.sum(preds == labels.data)
        valid_gt.extend(labels.data.cpu().numpy().tolist())
    valid_acc = correct_v / len(valid_data)
    valid_gt = np.array(valid_gt)  # for confusion matrix later, don't shuffle valid!
    print(f'No train yet, Valid: {valid_acc:0.4f} = {correct_v} / {len(valid_data)}')
    # ---------------------------- now let's train! -------------------------- #

    # Store results for *this* run here.
    valid_accs = [valid_acc]
    valid_preds_best = None
    best_epoch = -1
    best_vacc = -1
    train_accs = []

    for epoch in range(1, args.epochs+1):
        # Train for 1 epoch. Don't forget to switch train mode and zero grads!
        model_ft.train()
        correct_t = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer_ft.zero_grad()
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            _, preds = torch.max(outputs, 1)
            correct_t += torch.sum(preds == labels.data)
        train_acc = correct_t / len(train_data)
        train_accs.append(train_acc)

        # Evaluate. Switch to eval, no gradient propagation.
        valid_preds = []
        model_ft.eval()
        correct_v = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            with torch.no_grad():
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
            correct_v += torch.sum(preds == labels.data)
            valid_preds.extend(preds.cpu().numpy().tolist())
        valid_acc = correct_v / len(valid_data)

        # Statistics, etc.
        if valid_acc > best_vacc:
            best_vacc = valid_acc
            best_epoch = epoch
            valid_preds_best = np.array(valid_preds)
        print(f'Epoch {epoch:03d}, Train: {train_acc:0.4f}, Valid: {valid_acc:0.4f}')
        valid_accs.append(valid_acc)

    # Evaluation. We can use the best valid predictions and just test?
    best_bacc = balanced_accuracy_score(valid_gt, valid_preds_best) # shapes are: (n_valid,)
    best_cf = confusion_matrix(valid_gt, valid_preds_best)
    print(f'\n[Done with training]')
    print(f'Best epoch / v_acc: {best_epoch}, {best_vacc:0.3f}')
    print(f'Balanced accuracy score: {best_bacc:0.4f}')
    print(best_cf)

    # ------------------------ Store Results/Info and Model Files ---------------------- #
    results_info = {
        'train_accs': train_accs,
        'valid_accs': valid_accs,
        'balanced_acc': best_bacc,
        'conf_matrix': best_cf,
        'feature_extract': feature_extract,
        'RESKIN_EXP': RESKIN_EXP,
        'CLASSES': CLASSES,
    }
    info_save = {
        'model_sd': model_ft.state_dict(),
        'optim_sd': optimizer_ft.state_dict(),
        'results_info': results_info,
    }
    count = len([x for x in os.listdir('results')
        if '.pth' in x and 'reskin_img_classifier' in x])
    savepath = join('results', f'reskin_img_classifier_{str(count).zfill(2)}.pth')
    torch.save(info_save, savepath)
    print(f'See the PyTorch model/optim state dicts here with other stats:\n\t{savepath}')


def classify_reskin(cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, n_folds=1,
        debug_print=True, gradually_increase=False, test_drift=0):
    """Classify ReSkin data from physical data collection.

    Capital letters = Train, lowercase = test. :) Edit: I changed this, too error prone.

    So for now we'll just use 4 classes (the last 2 are grasping 1 and 2 fabrics).
    Also, for each fold, we should have a way of quantifying the gain we get w.r.t the
    training data size.
    """
    stats = defaultdict(list)
    stats['cmat'] = np.zeros((4,4))

    for i in range(n_folds):
        print(f'\n|------------------------- Fold {i+1} of {n_folds} --------------------------|')
        # Randomize train / test for this fold. CAREFUL! Do not override X, Y, x, y.
        # NOTE(daniel): data is NOT YET normalized!
        # NOTE(daniel): if testing drift, we use additional `info` with data indices.
        X_train_orig, y_train_orig, X_test_orig, y_test_orig, info = U.get_dataset_from_dir_names(
                cloth0_dir_names, cloth1_dir_names, cloth2_dir_names,
                n_epis=args.n_epis, n_train=args.n_train, test_drift=args.test_drift)
        if debug_print:
            print(f'Train (X,Y): {X_train_orig.shape}, {y_train_orig.shape}')
            print(f'Test  (x,y): {X_test_orig.shape}, {y_test_orig.shape}')
            print(f'Total train/test: {len(y_train_orig) + len(y_test_orig)}')
            U.print_class_counts(y_train_orig, data_type='train')
            U.print_class_counts(y_test_orig, data_type='test')

        if args.method == 'rf':
            clf = RandomForestClassifier(max_depth=None, random_state=0)
        elif args.method == 'knn':
            clf = neighbors.KNeighborsClassifier(10, weights="distance")

        if test_drift == 1:
            # --------------------------------------------------------------------- #
            # Use `info` from above, which is a list of tuples:
            #   [(length, dirname)_1, ..., (length, dirname)_k ]
            # Note that these might have some mixing if 0, 1, 2 were collected one after
            # the other, or all 0s, then all 1s, etc. There's a lot of nuance here.
            # Sashank (for the franka norub data from mid-Feb) did all 0s, all 1s, all 2s.
            # --------------------------------------------------------------------- #
            print('|------| Testing drift, fix train, check on later episodes |------|')
            count = X.shape[0]
            test_lens = [tup[0] for tup in info]
            test_dirs = [tup[1] for tup in info]

            # Normalize, then fit based on the FULL training data.
            X_train, X_test = U.normalize_data(X_train_orig, X_test_orig)
            clf.fit(X_train, y_train_orig.ravel())
            y_pred = clf.predict(X_test)

            # Measure accuracies over time. For now, not using balanced due to:
            #   UserWarning: y_pred contains classes not in y_true
            # Which makes sense given the structure of data Sashank collected.
            accs_time = []
            start = 0
            for (tl,td) in zip(test_lens,test_dirs):
                end = start + tl
                y_this_epis = np.squeeze(y_test_orig[start:end])
                num_equal = np.count_nonzero(y_this_epis == y_pred[start:end])
                accuracy = float(num_equal) / (end-start)
                #print(start, end, accuracy)
                if accuracy < 0.8:
                    print(f'Warning: check {td}, acc: {accuracy:0.4f}')
                accs_time.append(accuracy)
                start = end
            accs_time = np.array(accs_time)
            print('Here are the results from testing drift:')
            print(f'  {accs_time}')

            # Might be easier to save a quick plot.
            nrows, ncols = 1, 1
            fig, ax = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
            xs = np.arange(len(test_lens))
            xlabel = f'Data collection over time'
            ylabel = f'Test accuracy'
            label = f'Low: {np.min(accs_time):0.3f}, High: {np.max(accs_time):0.3f}'
            titlestr = f'Testing Drift. X_train: {X_train.shape}'
            pltname = f'test_drift_{args.test_drift}.png'
            plt.plot(xs, accs_time, lw=lw, label=label)
            plt.xlabel(xlabel, size=xsize)
            plt.ylabel(ylabel, size=ysize)
            plt.xticks(fontsize=xticksize)
            plt.yticks(fontsize=yticksize)
            plt.title(titlestr, size=titlesize)
            plt.ylim([0.40,1.01])
            plt.legend(loc="best", ncol=1, prop={'size': legendsize})
            plt.tight_layout()
            plt.savefig(pltname)
            print(f'Plot saved at: {pltname}')

        elif test_drift == 2:
            # --------------------------------------------------------------------- #
            # Use `info` from above, which is a list of tuples:
            #   [(length, dirname)_1, ..., (length, dirname)_k ]
            # Now we're using an entirely separate dataset for checking drift.
            # Lots of the code is duplicate, can figure out how to simplify later.
            # Here we can check both the normal test set AND the new data entirely.
            # Also now I realize, we actually can and probably should randomize the
            # order of train / valid episodes for this one.
            # --------------------------------------------------------------------- #
            print('|------| Testing drift, fix train, check on later episodes |------|')

            # In this case the TESTDRIFT just has 1 class.
            print(f'\nNow what happens on data from: {TESTDRIFT}')
            cloth0_dir_names, _, _, = U.get_cloth_dir_names(p0=TESTDRIFT)
            print(f'  number of directories (episodes): {len(cloth0_dir_names)}')
            _, _, X_test_d, y_test_d, info = U.get_raw_data(train_dir_names=[],
                    test_dir_names=cloth0_dir_names, new_collect=True, get_images=False)
            test_lens = [tup[0] for tup in info]
            test_dirs = [tup[1] for tup in info]
            if debug_print:
                print(f'  (new data) X_test_d: {X_test_d.shape}')
                print(f'  (new data) y_test_d: {y_test_d.shape}')
                print(f'  (new data) test_lens: {test_lens}')

            # Normalize, then fit based on the full training data.
            X_train, X_test = U.normalize_data(X_train_orig, X_test_orig)
            clf.fit(X_train, y_train_orig.ravel())
            y_pred = clf.predict(X_test)  # predict on test set from SAME data batch
            score = balanced_accuracy_score(y_test_orig, y_pred)
            best_cf = confusion_matrix(y_test_orig, y_pred)
            print(f'[Test Drift] On the same data, balanced acc: {score:0.4f}')
            print(best_cf)
            first2_correct = best_cf[0,0]+best_cf[1,1]
            first2_acc = first2_correct / (first2_correct + best_cf[0,1]+best_cf[1,0])
            print(f'Just first two:: {first2_acc:0.3f}')

            # The same normalization, use same original data `X` except now with different test.
            # Note that we don't re-train, we have the previously trained classifier.
            debugging, X_test_d = U.normalize_data(X_train_orig, X_test_d)
            assert np.array_equal(X_train, debugging)
            y_pred_d = clf.predict(X_test_d)  # predict on a fully different test set.
            y_pred_d = y_pred_d[:,None]  # reshape (n,) --> (n,1)
            print('\nCounts of predictions in y_pred:')
            for i in range(4):
                print(f'  {i}: {np.sum(y_pred_d == i)}')
            score = balanced_accuracy_score(y_test_d, y_pred_d)
            best_cf = confusion_matrix(y_test_d, y_pred_d)
            print(f'\n[Test Drift] On the new data, balanced acc: {score:0.4f}')
            print(best_cf)
            num_correct = np.sum(y_pred_d == y_test_d)
            print(f'[Test Drift] Raw acc: {(num_correct / len(y_test_d)):0.4f}')

            # One way to check drift.
            print('\nCan check normalization statistics, see if new data is off more:')
            print(f'X_test (same data) mean: {np.mean(X_test, axis=0)}')
            print(f'X_test (diff data) mean: {np.mean(X_test_d, axis=0)}')

            # Measure accuracies over time. Here use `y_pred_d` NOT `y_pred`!!
            # Note a bit annoying but now we also need to squeeze `y_pred_d``.
            accs_time = []
            start = 0
            for (tl,td) in zip(test_lens,test_dirs):
                end = start + tl
                y_this_epis = np.squeeze(y_test_d[start:end])  # ground-truth
                y_this_pred = np.squeeze(y_pred_d[start:end])  # prediction
                num_equal = np.count_nonzero(y_this_epis == y_this_pred)
                assert num_equal <= (end-start)
                accuracy = float(num_equal) / (end-start)
                #if accuracy < 0.5:
                #    print(f'Warning: check {td}, acc: {accuracy:0.4f}')
                #print(td, num_equal, end-start, y_this_epis, y_this_pred)
                accs_time.append(accuracy)
                start = end
            accs_time = np.array(accs_time)
            print('\nHere are the results from testing drift:')
            print(f'  {accs_time} (avg: {np.mean(accs_time):0.3f})')

            # Might be easier to save a quick plot.
            nrows, ncols = 1, 1
            fig, ax = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
            xs = np.arange(len(test_lens))
            xlabel = f'Data collection over time'
            ylabel = f'Test accuracy'
            label = f'Low: {np.min(accs_time):0.3f}, High: {np.max(accs_time):0.3f}'
            titlestr = f'Testing Drift. X_train: {X_train.shape}'
            pltname = f'test_drift_{args.test_drift}.png'
            plt.plot(xs, accs_time, lw=lw, label=label)
            plt.xlabel(xlabel, size=xsize)
            plt.ylabel(ylabel, size=ysize)
            plt.xticks(fontsize=xticksize)
            plt.yticks(fontsize=yticksize)
            plt.title(titlestr, size=titlesize)
            plt.ylim([0.40,1.01])
            plt.legend(loc="best", ncol=1, prop={'size': legendsize})
            plt.tight_layout()
            plt.savefig(pltname)
            print(f'Plot saved at: {pltname}')

        if gradually_increase:
            # --------------------------------------------------------------------- #
            # Before we do this, we'll keep using subsets of X to print. We need to
            # shuffle, though, since otherwise we'd get some data imbalance. We were
            # not doing earlier but that's because kNN does not require shuffling.
            # NOTE(daniel): if we normalize X and x according to full data stats, then
            # we can get really good results even with just 37 training data points!
            # --------------------------------------------------------------------- #
            print('|------| Gradually increasing train data for fixed test |------|')
            percentages = [0.0002, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.0]
            count = X_train_orig.shape[0]
            acc_all = []

            for seed in range(10):
                print(f'seed {seed}')
                acc_seed = []
                for perc in percentages:
                    num_data = int(perc * count)
                    indices = np.random.permutation(count)[:num_data]  # random indices
                    X_subset_train = X_train_orig[indices, :]
                    Y_subset_train = y_train_orig[indices]
                    # Normalize, then fit  based on a subset of the data. Full test = x.
                    X_subset_train, X_subset_test = U.normalize_data(X_subset_train, X_test_orig)
                    clf.fit(X_subset_train, Y_subset_train.ravel())
                    y_pred = clf.predict(X_subset_test)
                    score = balanced_accuracy_score(y_test_orig, y_pred)
                    best_cf = confusion_matrix(y_test_orig, y_pred)
                    print(f'  {perc:0.4f}, X: {X_subset_train.shape}, score: {score:0.4f}')
                    #print(f'{best_cf}\n')
                    acc_seed.append(score)
                acc_all.append(acc_seed)

            # (num_seeds, num_percentages)
            acc_all = np.array(acc_all)
            print('Here are the results from gradually increasing data size:')
            print(f'Percentages: {percentages}')
            print(f'  Mean: {np.mean(acc_all, axis=0)}')
            print(f'  Std: {np.std(acc_all, axis=0)}\n')

        # Scipy / sklearn uses a consistent API, fortunately. Train on FULL data.
        X, x = U.normalize_data(X_train_orig, X_test_orig)
        clf.fit(X, y_train_orig.ravel())
        y_pred = clf.predict(x)
        score = balanced_accuracy_score(y_test_orig, y_pred)
        best_cf = confusion_matrix(y_test_orig, y_pred)
        print(f'\n[Full data]\nBalanced accuracy score: {score:0.4f}')
        print(best_cf)
        stats['score'].append(score)
        stats['cmat'] += best_cf  # keep ADDING so we can divide

    print(f'\n-------------------------------------------------------')
    print(f'Done over {n_folds} different folds.')
    print('Score: {:0.4f} +/- {:0.2f}'.format(
            np.mean(stats['score']), np.std(stats['score'])))
    print('Avg confusion matrix:\n{}'.format(stats['cmat'] / n_folds))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, default='knn')
    p.add_argument('--n_folds', type=int, default=1)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gradually_increase', action='store_true', default=False,
        help='Only supported for kNN / RF so far.')
    p.add_argument('--test_drift', type=int, default=0,
        help='Only for kNN. 1 = same data dir, 2 = different data dir')
    args = p.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert not (args.gradually_increase and args.test_drift != 0), 'Test one, not both.'

    # Load all ReSkin data paths. This is train and test for all classes.
    cloth0_dir_names, cloth1_dir_names, cloth2_dir_names = U.get_cloth_dir_names(
        p0=CLASSES[0], p1=CLASSES[1], p2=CLASSES[2])
    print(f'  len cloth0_dir_names: {len(cloth0_dir_names)}')
    print(f'  len cloth1_dir_names: {len(cloth1_dir_names)}')
    print(f'  len cloth2_dir_names: {len(cloth2_dir_names)}')
    args.n_epis = n_epis = len(cloth0_dir_names)

    # Note for drift, use smaller values so we have more test.
    if args.test_drift == 0:
        args.n_train = n_train = int(n_epis * 0.80)  # the usual case
    elif args.test_drift == 1:
        args.n_train = n_train = int(n_epis * 0.50)  # same data, hence decrease
    elif args.test_drift == 2:
        args.n_train = n_train = int(n_epis * 0.80)  # can be in-between, etc.
    else:
        raise ValueError(args.test_drift)

    print(f'About to classify ReSkin, n_train / n_epis: {n_train} / {n_epis}')
    print(f'Method: {args.method}')

    # Next let's classify. Each classification involves a different train/test fold.
    if args.method in ['knn', 'rf']:
        classify_reskin(cloth0_dir_names,
                        cloth1_dir_names,
                        cloth2_dir_names,
                        n_folds=args.n_folds,
                        gradually_increase=args.gradually_increase,
                        test_drift=args.test_drift)
    elif args.method in ['cnn']:
        classify_reskin_img(cloth0_dir_names,
                            cloth1_dir_names,
                            cloth2_dir_names,
                            args=args)
    else:
        raise ValueError(args.method)