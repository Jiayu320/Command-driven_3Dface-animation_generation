import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
import scipy.io as sio


class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.AU_data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.AU_data_root, self.opt.imgs_dir)
        self.FLAME_dir = self.opt.FLAME_data_root
        # print('self.FLAME_dir:'+ self.FLAME_dir)
        self.is_train = self.opt.mode == "train"

        # load images path 
        filename = self.opt.train_csv if self.is_train else self.opt.test_csv
        self.imgs_name_file = os.path.join(self.opt.AU_data_root, filename)
        self.imgs_path = self.make_dataset()
        
        # load FLAME.mat dicitionary 
        #self.FLAMEs_name_file = os.path.join(self.opt.FLAME_data_root, filename)
        self.FLAMEmats_root = self.get_FLAMEmats_root()

        # load AUs dicitionary 
        aus_pkl = os.path.join(self.opt.AU_data_root, self.opt.aus_pkl)
        self.aus_dict = self.load_dict(aus_pkl)

    def make_dataset(self):
        return None
    
    def get_FLAMEmats_root(self):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)
    
    def get_FLAME_by_path(self, FLAMEmat_root):
        assert os.path.isfile(FLAMEmat_root), "Cannot find FLAME.mat file: %s" % FLAMEmat_root
        # 读取matlab数据
        info = sio.loadmat(FLAMEmat_root)
        pose_para = info['pose'].T.astype(np.float32)
        
        exp_para = info['exp'].astype(np.float32)
        
        exp_re = np.reshape(exp_para,(1,-1))
        pose_re = np.reshape(pose_para,(1,-1))
        pose_re = pose_re[:, 3:]
        
        FLAME_feature = np.hstack((exp_re.flatten(), pose_re.flatten()))
        return FLAME_feature
    
    def show_FLAME_feature(self, FLAMEmat_root):
        import matplotlib.pyplot as plt
        sha_exp_pos = self.get_FLAME_by_path(FLAMEmat_root)
        plt.matshow(sha_exp_pos, cmap=plt.get_cmap('Blues'))
        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度
        for edge, spine in plt.gca().spines.items():
            spine.set_visible(False)
            
        path_parts = FLAMEmat_root.split(os.sep)
        desired_part = path_parts[-2]
        save_path = 'FLAME_feature_fig/' + desired_part + '_feature.png'
        
        # 检查 fig 文件夹是否存在
        if not os.path.exists('FLAME_feature_fig'):
            # 如果不存在，则创建
            os.makedirs('FLAME_feature_fig')
            print("Folder 'FLAME_feature_fig' created.")
        
        plt.gca().set_xticks([], minor=True)
        plt.gca().set_yticks([], minor=True)
        
        plt.savefig(save_path)
        print("Please cheak the results in" + save_path)

    def get_aus_by_path(self, img_path):
        return None

    def __len__(self):
        return len(self.imgs_path)





    







