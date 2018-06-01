#importing dependencies

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import sys
import math
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

import skimage.morphology
from skimage.morphology import square, dilation, erosion


def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    '''

    '''

    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]] #
    indices = []
    values = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            #p index changed
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)

            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN > 1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)

    shape = [height, width, 1]

    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense.reshape((shape[0], shape[1], 1))



def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    '''
    Gives us the blot of pixels which are used to depict a heatmap for the pose
    '''
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i**2+j**2))
            if r + i >= 0 and r + i < height and c + j >= 0 and c + j < width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)

    return indices, values



def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    '''
    Gives us all the pixel indices used to get the sparse pose
    '''
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            #TODO: Changed here
            r = p[0][1]
            c = p[0][0]
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape



def _returnPose(indices, shape):
    '''
    Converts the sparse poses Pose images of the shape height * width * no.of heat maps
    '''
    height, width, channel = shape
    pose = np.zeros((height, width, channel))
    for ind in indices:
        pose[ind[0], ind[1], ind[2]] = 1

    return pose



def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = values[i]
    return dense



def _get_valid_peaks(all_peaks, subsets):
    '''
    Gives us the valid pose peaks from the possibility set of 18 poses
    '''
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            if score > valid_score:
                valid_idx = i
                valid_score = score

        if valid_idx >= 0:
            peaks = []
            cand_id_list = subsets[valid_idx][:18]
            for ap in all_peaks:
                valid_p = []
                for p in ap:
                    if p[-1] in cand_id_list:
                        valid_p = p
                peaks.append(valid_p)
            return all_peaks
        else:
            return None
    except:
        return None



def strip_flip(filename):
    '''Removes _flip from an image name while making the pair list so that it matches the key for flip_pose_dict'''
    fileList = filename.split('_Flip')
    return fileList[0] + fileList[1]



def _format_data(folder_path, pairs, i, labels, all_peaks_dic, subsets_dic, out_filename, cnt):
    '''
    Completes one example point to be stored in the dataset. The example point is composed of the raw images,
    pose maskes, poses, etc.
    '''

    # Read the filename:
    img_path_0 = os.path.join(folder_path, pairs[i][0])
    img_path_1 = os.path.join(folder_path, pairs[i][1])

    id_0 = pairs[i][0].split('_')[0]
    id_1 = pairs[i][1].split('_')[0]

    image_raw_0 = np.asarray(Image.open(img_path_0))
    image_raw_1 = np.asarray(Image.open(img_path_1))

    height, width, channels = image_raw_0.shape

    #removing Flip from the image path names to match the keys in the pose_flip_dict
    if pairs[i][0].find('_Flip') != -1 and pairs[i][1].find('_Flip') != -1:
        image_key_0 = strip_flip(pairs[i][0])
        image_key_1 = strip_flip(pairs[i][1])
    else:
        image_key_0 = pairs[i][0]
        image_key_1 = pairs[i][1]

    #checking that the all_peaks for the image exists
    if (all_peaks_dic is not None) and (image_key_0 in all_peaks_dic) and (image_key_1 in all_peaks_dic):

        #POSE 0
        peaks = _get_valid_peaks(all_peaks_dic[image_key_0], subsets_dic[image_key_0])

        indices_r4_0, values_r4_0, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        pose_r4_0 = _returnPose(indices_r4_0, shape)

        indices_r8_0, values_r8_0, shape = _getSparsePose(peaks, height, width, 18, radius=8, mode='Solid')
        pose_r8_0 = _returnPose(indices_r8_0, shape)

        pose_mask_r4_0 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r8_0 = _getPoseMask(peaks, height, width, radius=8, mode='Solid')


        ## Pose 1
        peaks = _get_valid_peaks(all_peaks_dic[image_key_1], subsets_dic[image_key_1])

        indices_r4_1, values_r4_1, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        pose_r4_1 = _returnPose(indices_r4_1, shape)

        indices_r8_1, values_r8_1, shape = _getSparsePose(peaks, height, width, 18, radius=8, mode='Solid')
        pose_r8_1 = _returnPose(indices_r8_1, shape)

        pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r8_1 = _getPoseMask(peaks, height, width, radius=8, mode='Solid')


    else:
        return False

    group_name = 'Data_{}'.format(cnt)
    with h5py.File(out_filename, 'a') as f:
        g = f.create_group(group_name)
        g.create_dataset('image_raw_0', data = image_raw_0, compression = 'gzip')
        g.create_dataset('image_raw_1', data = image_raw_1, compression = 'gzip')

        #NOTE: Sending data with radius = 8 for better mask
        g.create_dataset('pose_mask_r8_0', data = pose_mask_r8_0.astype(np.int64), compression = 'gzip')
        g.create_dataset('pose_mask_r8_1', data = pose_mask_r8_1.astype(np.int64), compression = 'gzip')


        g.create_dataset('pose_r8_0', data = pose_r8_0, compression = 'gzip')
        g.create_dataset('pose_r8_1', data = pose_r8_1, compression = 'gzip')

    return True



def _get_outdata_filename(out_dir, split_name):
    '''Gives output file name for the test/train hdf5 file'''
    output_filename = 'DF_{}.hdf5'.format(split_name)
    return os.path.join(out_dir, output_filename)



def _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip,
                                           labels, labels_flip, dataset_dir, pose_peak_path = None,
                                           pose_sub_path = None, pose_peak_path_flip = None,
                                           pose_sub_path_flip = None, max_cnt = np.inf):

    """Converts the given pairs to a HDF5 file format after getting poses, masks etc.
    Args:
        out_dir: where to store the resultant HDF5 format
        split_name: The name of the dataset, either 'train' or 'validation'.
        split_name_flip: if flipping is used with training as well. Deactivated if None
        pairs: A list of image name pairs.
        labels: label list to indicate positive(1) for same person pairs or (0) for diff person pairs
        dataset_dir: The directory from where to get the images.
        pose paths: paths to the poses and sub_poses for all_dict and flip_dict
        max_cnt: limit the number of samples copied in the hdf5 file
    """

    if split_name_flip is None:
        USE_FLIP = False
    else:
        USE_FLIP = True
        num_pairs_flip = len(pairs_flip)


    assert split_name in ['train', 'test', 'test_samples', 'test_seq']
    num_pairs = len(pairs)


    #get the folder from where to derive images
    folder_path = _get_folder_path(dataset_dir, split_name)

    if USE_FLIP:
        folder_path_flip = _get_folder_path(dataset_dir, split_name_flip)


    # Load pose pickle file
    all_peaks_dic = None
    subsets_dic = None
    all_peaks_dic_flip = None
    subsets_dic_flip = None

    with open(pose_peak_path, 'rb') as f:
        all_peaks_dic = pickle.load(f, encoding='latin1')

    with open(pose_sub_path, 'rb') as f:
        subsets_dic = pickle.load(f, encoding='latin1')

    if USE_FLIP:
        with open(pose_peak_path_flip, 'rb') as f:
            all_peaks_dic_flip = pickle.load(f, encoding='latin1')
        with open(pose_sub_path_flip, 'rb') as f:
            subsets_dic_flip = pickle.load(f, encoding='latin1')


    out_filename = _get_outdata_filename(out_dir, split_name)
    cnt = 0 #counts the number of entry pairs

    if USE_FLIP:
        print("Processing flip data...")
        for i in range(num_pairs_flip):

            Ok = _format_data(folder_path_flip, pairs_flip, i, labels_flip, all_peaks_dic_flip, subsets_dic_flip, out_filename, cnt)

            if (Ok == False):
                continue

            cnt += 1
            if cnt % 500 == 0:
                print("Saving flip data at count: {}".format(cnt))

            if cnt >= max_cnt:
                break

    print("Processing normal data...")
    for i in range(num_pairs):
        if cnt >= max_cnt:
            break

        Ok = _format_data(folder_path, pairs, i, labels, all_peaks_dic, subsets_dic, out_filename, cnt)

        if (Ok == False):
            continue

        cnt += 1
        if cnt % 500 == 0:
            print("Saving data at count: {}".format(cnt))


    with open(os.path.join(out_dir,'total_pairs.txt'),'w') as f:
        f.write('cnt:%d' % cnt)



def _get_folder_path(dataset_dir, split_name):
    '''
    Gets the appropriate input image folder set from the dataset_dir based on the split name
    '''
    if split_name == 'train':
        folder_path = os.path.join(dataset_dir, 'filted_up_train')
    elif split_name == 'train_flip':
        folder_path = os.path.join(dataset_dir, 'filted_up_train_flip')
    elif split_name == 'test':
        folder_path = os.path.join(dataset_dir, 'filted_up_test')
    elif split_name == 'test_samples':
        folder_path = os.path.join(dataset_dir, 'filted_up_test_samples')
    elif split_name == 'test_seq':
        folder_path = os.path.join(dataset_dir, 'filted_up_test_seq')

    assert os.path.isdir(folder_path)

    return folder_path



def _get_image_file_list(dataset_dir, split_name):
    '''
    Returns a list of all the jpg or png files in the input image folder
    '''

    folder_path = _get_folder_path(dataset_dir, split_name)

    if split_name == 'train' or split_name == 'train_flip' or split_name == 'test':
        filelist = sorted(os.listdir(folder_path))
    else:
        raise IOError('Split_name not in accepted list')

    # Remove non-jpg files
    valid_filelist = []
    for i in range(len(filelist)):
        if filelist[i].endswith('.jpg') or filelist[i].endswith('.png'):
            valid_filelist.append(filelist[i])

    return valid_filelist



def _get_train_all_pn_pairs(input_dir, out_dir, split_name = 'train', augment_ratio = 1, mode = 'same_diff_cam'):

    """Returns a list of pair image filenames.
    Args:
        input_dir: A directory containing person images. Using this, it will make pose pairs of a person.
        out_dir: where to store the image pairs with each of their's file name
        split_name: 'train', 'train_flip', 'test'

    Returns:
        p_pairs: A list of positive pairs.
        n_pairs: A list of negative pairs. (Won't be using here)
    """

    assert split_name in {'train', 'train_flip', 'test', 'test_samples', 'test_seq', 'all'}

    p_pairs_path = os.path.join(out_dir, 'p_pairs_'+ split_name +'.p')
    n_pairs_path = os.path.join(out_dir, 'n_pairs_'+ split_name +'.p')
    count_path = os.path.join(out_dir, 'count_'+ split_name +'.p')

    #if p pairs have already been made, skip making another one
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path,'rb') as f:
            p_pairs = pickle.load(f, encoding='latin1')
        with open(n_pairs_path,'rb') as f:
            n_pairs = pickle.load(f, encoding='latin1')

        print("Pickle files for {} loaded from existing folders".format(split_name))
        print('p_pairs length: {}'.format(len(p_pairs)))
        print('n_pairs length: {}'.format(len(n_pairs)))

    else:
        filelist = _get_image_file_list(dataset_dir, split_name)
        p_pairs = []
        n_pairs = []

        if split_name == 'test_seq':
            for i in range(0, len(filelist)):
                for j in range(len(filelist)):
                    p_pairs.append([filelist[i],filelist[j]])

        elif mode == 'same_diff_cam':
            for i in range(0, len(filelist)):
                names = filelist[i].split('_')
                id_i = names[0]
                for j in range(i+1, len(filelist)):
                    names = filelist[j].split('_')
                    id_j = names[0]

                    if id_j == id_i:
                        p_pairs.append([filelist[i],filelist[j]])
                        p_pairs.append([filelist[j],filelist[i]])

                    #creating n pairs as well, but we won't be using it
                    elif j%2000 == 0 and id_j != id_i:  # limit the neg pairs
                        n_pairs.append([filelist[i],filelist[j]])


        p_pairs = p_pairs * augment_ratio
        random.shuffle(n_pairs)
        n_pairs = n_pairs[:len(p_pairs)]

        num_p_pairs = len(p_pairs)
        num_n_pairs = len(n_pairs)

        print("Pickle files for {} saved to {}".format(split_name, out_dir))
        print('p_pairs length: {}'.format(num_p_pairs))
        print('n_pairs length: {}'.format(num_n_pairs))

        with open(p_pairs_path,'wb') as f:
            pickle.dump(p_pairs,f)
        with open(n_pairs_path,'wb') as f:
            pickle.dump(n_pairs,f)

        with open(count_path,'wb') as f:
            pickle.dump("[Num_p_pairs, num_n_pairs]: \n",f)
            pickle.dump([num_p_pairs, num_n_pairs],f)

    return p_pairs, n_pairs


def run_one_pair_rec(dataset_dir, out_dir, split_name):

    print("In run_one_pair_rec")

    #creating the required directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Directory made: {}'.format(out_dir))

    #we'll train with the train and train_flip folder
    if split_name.lower()=='train':

        #================ Prepare training set ================

        #All poses are stored in the pickle files in the input folder, seperately for all images and flip images

        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        pose_peak_path_flip = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion_Flip.p')
        pose_sub_path_flip = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion_Flip.p')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name = split_name,
                                                    augment_ratio=1,
                                                    mode='same_diff_cam')
        p_labels = [1]* len(p_pairs)
        n_labels = [0]* len(n_pairs) #0 for pairs which are not of the same person

        pairs = p_pairs
        labels = p_labels

        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        split_name_flip = 'train_flip'
        p_pairs_flip, n_pairs_flip = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name = split_name_flip,
                                                    augment_ratio = 1,
                                                    mode = 'same_diff_cam')
        p_labels_flip = [1]*len(p_pairs_flip)
        n_labels_flip = [0]*len(n_pairs_flip)

        pairs_flip = p_pairs_flip
        labels_flip = p_labels_flip
        combined = list(zip(pairs_flip, labels_flip))
        random.shuffle(combined)
        pairs_flip[:], labels_flip[:] = zip(*combined)

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels,
                                               labels_flip, dataset_dir,
                                               pose_peak_path = pose_peak_path,
                                               pose_sub_path = pose_sub_path,
                                               pose_peak_path_flip = pose_peak_path_flip,
                                               pose_sub_path_flip = pose_sub_path_flip)

        print('\n Train convert Finished !')

    if split_name.lower()=='test':

        # ================ Prepare test set ================
        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                  split_name = split_name,
                                                  augment_ratio = 1,
                                                  mode= 'same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels,
                                               labels_flip, dataset_dir,
                                               pose_peak_path = pose_peak_path,
                                               pose_sub_path = pose_sub_path)

        print('\nTest convert Finished !')


if __name__ == '__main__':
    cwd = os.getcwd()
    dataset_folder = sys.argv[1] #'DF_img_pose'
    split_name = sys.argv[2]   ## 'train' or 'test'

    dataset_dir = os.path.join(cwd, dataset_folder)
    out_dir = os.path.join(cwd, 'DF_' + split_name +'_data')

    run_one_pair_rec(dataset_dir, out_dir, split_name)
