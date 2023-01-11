# Image Classifier

Use this as a baseline for the tactile classifier.

Installation / setup:

```
conda create --name reskin-image python=3.6 -y
conda activate reskin-image
conda install ipython -y 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install scikit-learn
pip install matplotlib
pip install moviepy
pip install opencv-python
```

Use this if CUDA 11.3 isn't supported on your machine / GPU:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
```


## Data Formatting and Code Usage.

Assumes data is of the form:

```
seita@takeshi:/data/seita/softgym_ft/franka_norub_folded_random $ ls -lh
total 12K
drwxrwxr-x 47 seita seita 4.0K Feb 13 11:49 0cloth_norub_auto
drwxrwxr-x 47 seita seita 4.0K Feb 13 11:49 1cloth_norub_auto
drwxrwxr-x 47 seita seita 4.0K Feb 13 11:49 2cloth_norub_auto
seita@takeshi:/data/seita/softgym_ft/franka_norub_folded_random $
```

This has 3 subdirectories, each of which contains directories corresponding to
different data collections (we sometimes call these "episodes" or "trials"). The
class is based on the gripper touching and getting 0, 1, or 2 cloth layers.
There is also a 4th class prevalent from *all* these subdirectories, indicating
when the gripper is open (this is the class indexed as 0 in our code).

Train the following 4-way classifiers, which can be with neural nets (currently
pre-trained ResNets), or kNN, RFs, or others that can be imported from scikit:

```
python scripts/reskin_classify.py --method cnn
python scripts/reskin_classify.py --method knn
python scripts/reskin_classify.py --method rf
```

For kNN and RFs, we can optionally test with gradually increasing the size of
the training data (while keeping validation fixed) or with testing drift, where
we fix the training data, and then test performance as a function of later
(subsequent) episodes. For example:

```
python scripts/reskin_classify.py --method knn --gradually_increase
python scripts/reskin_classify.py --method knn --test_drift 1
python scripts/reskin_classify.py --method knn --test_drift 2
```

See the file for more arguments. *Be careful with drift, because that depends a
lot on the order the data was collected, and this could be an issue if all the
0s were collected before the 1s, etc.* Use 1 if trying to deal with the same
data, and 2 if dealing with a separate data (must also specify that in the
python script).

Utility scripts are in: `scripts/reskin_utils.py`.

Note on comparisons: there will be more than 10X tactile data compared to
image-based data due to frequencies of data. If we want to compare tactile vs
image based classifiers in a little more detail, we can subsample the tactile
data.


## Results

TODO


