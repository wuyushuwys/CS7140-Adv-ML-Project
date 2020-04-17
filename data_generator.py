from scipy.io import loadmat
from glob import glob
import numpy as np
import os
from tqdm import tqdm
import argparse


def split_signal_single(data):
    l = len(data)
    splited_data = []
    l -= 128*5 # skip first 5 seconds as well as the first clip in the Ext_annotation, Data sampled in 128Hz
    baseline = data[:-l].reshape(-1,128)[1:,:].mean(0)
    t = 1
    while l-t*128>0:
        splited_data.append(data[-l:-l+128*t]-baseline)
        l -= 128*t
        
    return splited_data


def split_signal_multi(data):
    l, c = data.shape
    splited_data = []
    l -= 128*5 # skip first 5 seconds as well as the first clip in the Ext_annotation, Data sampled in 128Hz
    baseline = data[:128*5, :].reshape(-1, 128, c)[1:,:, :].mean(0)
    t = 1
    while l-t*128>0:
        splited_data.append(data[-l:-l+128*t, :]-baseline)
        l -= 128*t
    return splited_data

def main(filepath, folder):
    files = glob(filepath+"/*/*.mat")
    files.sort()


    print("Extracting data...")
    # GSR_SHAPE = []
    GSR_DATA = []
    EEG_DATA = []
    ECG_DATA = []
    ground_truth = []

    for f_idx, f in enumerate(tqdm(files)):
        if not any(ele in f for ele in ['09', '12', '21', '22', '23', '24', '33']):
            part_data = loadmat(f) # load participant's all preprocessed physiological data.
            _, num_videos =  part_data['joined_data'].shape
            video_head = 0
            num_videos -= 4


            for v_idx in range(video_head, num_videos):
                phys_data = part_data['joined_data'][0, v_idx] # the preprocessed physiological data of #idx video (not videoID)
                part_self_assessment = part_data['labels_selfassessment'][0, v_idx][0,:2]
                part_gsr_data = phys_data[:,16]
                part_eeg_data = phys_data[:, 0:14]
                part_ecg_data = phys_data[:, 14:16]
                split_gsr_data = split_signal_single(part_gsr_data)

                if part_self_assessment.sum()==0:
                    print("NULL self assessment in P{} video {}".format(f_idx+1, v_idx+1))
                GSR_DATA += split_gsr_data
                EEG_DATA += split_signal_multi(part_eeg_data)
                ECG_DATA += split_signal_multi(part_ecg_data)
                assert len(GSR_DATA) == len(EEG_DATA) == len(ECG_DATA)
                ground_truth += [part_self_assessment.tolist()] * len(split_gsr_data)

    print("Saving data in "+folder+" folder")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    np.save(folder+"/GSR_DATA" ,GSR_DATA)
    np.save(folder+"/EEG_DATA", EEG_DATA)
    np.save(folder+"/ECG_DATA", ECG_DATA)
    np.save(folder+"/ground_truth", ground_truth)
    print("Finish")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convolutional Neural Network')
    parser.add_argument('--filepath', type=str, default='data_preprocessed', help='data folder')
    parser.add_argument('--folder', type=str, default='processed_data', nargs='?', help='save dir')
    arg = parser.parse_args()
    
    main(arg.filepath, arg.folder)