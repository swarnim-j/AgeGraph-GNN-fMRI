from typing import Callable, Optional
import pandas as pd
import os
import nibabel as nib
import pickle
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import zscore
import torch
from torch_geometric.data import Data,InMemoryDataset
from random import randrange
import math
import zipfile
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import random

class Age_Dataset(InMemoryDataset):

    def __init__(self, root, dataset_name, dataset, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root, self.dataset_name, self.dataset = root, dataset_name, dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']
    
    def process(self):
        age_dataset = []
        for data in self.dataset:
            labels = data.y
            age = labels[1].item()
            if int(age) <= 2:
                data = Data(x=data.x, edge_index=data.edge_index, y=int(age))
                age_dataset.append(data)

        if self.pre_filter is not None:
            age_dataset = [data for data in age_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            age_dataset = [self.pre_transform(data) for data in age_dataset]

        data, slices = self.collate(age_dataset)
        print("saving path:", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class Brain_Connectome_State(InMemoryDataset):
    
    def __init__(self, root, dataset_name,n_rois, threshold,path_to_data,n_jobs,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.path_to_data,self.n_jobs = root, dataset_name,n_rois,threshold,path_to_data,n_jobs
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']
    
    def extract_from_3d_no(self,volume, fmri):
        """
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        """

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y


    def construct_adj_postive_perc(self,corr):
        """construct adjacency matrix from the given correlation matrix and threshold"""
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj_task(self,iid,target_path,volume):
        emotion_path = "tfMRI_EMOTION_LR.nii.gz"
        reg_path = "Movement_Regressors.txt"

        gambling_path = "tfMRI_GAMBLING_LR.nii.gz"
        
        language_path = "tfMRI_LANGUAGE_LR.nii.gz"

        motor_path = "tfMRI_MOTOR_LR.nii.gz"
        relational_path = "tfMRI_RELATIONAL_LR.nii.gz"

        social_path = "tfMRI_SOCIAL_LR.nii.gz"

        wm_path = "tfMRI_WM_LR.nii.gz"
        all_paths = [emotion_path,gambling_path,language_path,motor_path,relational_path,social_path,wm_path]
        data_list = []
        for y, path in enumerate(all_paths):
            try:
                image_path_LR = os.path.join(target_path, iid+"_"+path)
                reg_path = os.path.join(target_path, reg_path+"_"+reg_path)
                img = nib.load(image_path_LR)
                regs = np.loadtxt(reg_path)
                # regs_dt = np.loadtxt(regdt_path)
                fmri = img.get_fdata()
                Y = self.extract_from_3d_no(volume,fmri)
                start = 1
                stop = Y.shape[0]
                step = 1
                # detrending
                t = np.arange(start, stop+step, step)
                tzd = zscore(np.vstack((t, t**2)), axis=1)
                XX = np.vstack((np.ones(Y.shape[0]), tzd))
                B = np.matmul(np.linalg.pinv(XX).T,Y)
                Yt = Y - np.matmul(XX.T,B) 
                # regress out head motion regressors
                B2 = np.matmul(np.linalg.pinv(regs),Yt)
                Ytm = Yt - np.matmul(regs,B2) 

                # zscore over axis=0 (time)
                zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
                conn = ConnectivityMeasure(kind='correlation')
                fc = conn.fit_transform([Ytm])[0]
                zd_fc = conn.fit_transform([zd_Ytm])[0]
                fc *= np.tri(*fc.shape)
                np.fill_diagonal(fc, 0)

                # zscored upper triangle
                zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
                np.fill_diagonal(zd_fc, 0)
                corr = torch.tensor(fc + zd_fc).to(torch.float)
                A = self.construct_Adj_postive_perc(corr)
                edge_index = A.nonzero().t().to(torch.long)
            
                data = Data(x = corr, edge_index=edge_index, y = y)
                data_list.append(data)

                # os.remove(image_path_LR)
                # os.remove(reg_path)
            except:
                print("file skipped!") 
            
        return data_list

    def process(self):
        dataset = []
        target_path = self.target_dir
        
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_task)(iid,self.path_to_data,volume) for iid in tqdm(ids))
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])