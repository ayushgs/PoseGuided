# PoseGuided
Code for Pose Guided Image Generation implemented in PyTorch 0.4

# Requirements
Python 3.5+ <br />
PyTorch 0.4 <br />
TensorFlow 1.7+ (for TensorBoard) <br />
Skicit-Image <br />
Pickle <br />
H5PY <br />


# Data:


## Processed Data:

#### Final data
https://drive.google.com/open?id=1LSXIeoKhV0rvGhA0PJhojB73JhbZcfeN

Training data count: 127, 022

Test data: 18, 568


### P.S.
Check the path. Code assumes you are storing the DF_train_data and DF_test_data unzipped from the zip file in the cwd from where they are called to access.

# Training
From the code directory: ./run_train.sh

# Testing
From the code directory: ./run_test.sh

# Acknowledgments
This work is an extension of the [Pose Guided Person Generation Network](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf).

The data prep pipeline derives from the authors' [github code](https://github.com/charliememory/Pose-Guided-Person-Image-Generation).
