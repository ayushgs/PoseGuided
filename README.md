# PoseGuided
Code for Pose Guided Image Generation implemented in PyTorch 0.4

# Requirements
Python 3.5+
PyTorch 0.4
TensorFlow 1.7+ (for TensorBoard)
Skicit-Image
Pickle
H5PY


# Data:
## Raw Data:
Filtered and downsized train data: https://drive.google.com/open?id=1-cy7-uWDmtBeOfxddoN-HMZTjEXj_p-Z

## Processed Data:
Processed Data
To download complete training data, use the command:
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XhFe7pnH6NQJ7ariAfucLdDrKTrjs87a' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=1XhFe7pnH6NQJ7ariAfucLdDrKTrjs87a" -O DF_train_data.zip && rm -rf /tmp/cookies.txt

To download complete test set, use the command:
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=13F6ghjtmqCZHDnRLc33r4wTa1-urxDVF' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=13F6ghjtmqCZHDnRLc33r4wTa1-urxDVF" -O DF_test_data.zip && rm -rf /tmp/cookies.txt

Train and test data: https://drive.google.com/open?id=1IqR81O0ioDPQpIya0njyY2qFMDEKlO_X

10,000+ train data: https://drive.google.com/open?id=1_GinawCMN3wQjJDFvlzCxxDxjtwz5zLe

Final data: https://drive.google.com/open?id=1LSXIeoKhV0rvGhA0PJhojB73JhbZcfeN
Training data count: 127022
Test data: ~18k


### P.S.
Check the path. Code assumes you are storing the DF_train_data and DF_test_data unzipped from the zip file in the cwd from where they are called to access.

# Training
From the code directory: ./run_train.sh

# Testing
From the code directory: ./run_test.sh

