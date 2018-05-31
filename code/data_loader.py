import h5py
import numpy as np
import torch
from torch.utils import data
import torch

class Dataset(data.Dataset):
    '''Returns a customized sample of data'''

    def __init__(self, filename, total_data, to_nchw = 'True'):
        '''Initialization'''
        self.file = filename
        self.total = total_data
        self.to_nchw = to_nchw

    def __len__(self):
        '''Denotes the total number of samples'''
        return self.total

    def proc_rawimage(self, img, mean, norm):
        '''Processing the raw image. Here mean and norm are scalar values'''
        return (img - mean)/norm

    def hwc_to_chw(self, arr):
        return arr.transpose([2,0,1])

    def __getitem__(self, index):
        file = self.file
        group = 'Data_{}'.format(index)
        with h5py.File(file, 'r') as f_read:
            image_raw_0 = np.asarray(f_read[group].get('image_raw_0'), dtype = np.float32)
            image_raw_1 = np.asarray(f_read[group].get('image_raw_1'), dtype = np.float32)
            pose_r4_0 = np.asarray(f_read[group].get('pose_r4_0'), dtype = np.float32)
            pose_r4_1 = np.asarray(f_read[group].get('pose_r4_1'), dtype = np.float32)
            pose_mask_r4_0 = np.asarray(f_read[group].get('pose_mask_r4_0'), dtype = np.float32)
            pose_mask_r4_1 = np.asarray(f_read[group].get('pose_mask_r4_1'), dtype = np.float32)

        #processing
        proc_raw_0 = self.proc_rawimage(image_raw_0, 127.5, 127.5)
        proc_raw_1 = self.proc_rawimage(image_raw_1, 127.5, 127.5)
        proc_pose_0 = 2 * pose_r4_0 - 1
        proc_pose_1 = 2 * pose_r4_1 - 1

        #transforming to nchw format if required
        if self.to_nchw:
            proc_raw_0 = self.hwc_to_chw(proc_raw_0)
            proc_raw_1 = self.hwc_to_chw(proc_raw_1)
            proc_pose_0 = self.hwc_to_chw(proc_pose_0)
            proc_pose_1 = self.hwc_to_chw(proc_pose_1)
            pose_mask_r4_0 = self.hwc_to_chw(pose_mask_r4_0)
            pose_mask_r4_1 = self.hwc_to_chw(pose_mask_r4_1)


        return proc_raw_0, proc_raw_1, proc_pose_0, proc_pose_1, pose_mask_r4_0, pose_mask_r4_1
