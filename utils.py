
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from fnmatch import fnmatch
import os,re
import numpy as np

from typing import List

def load_mr_image(subj, ttype):
    suff = ttype.upper()
    if ttype == 't2': 
        suff = f'{suff}_reg'
    return nib.load(f'./data/small/{ttype}/{subj}-{suff}_fcm.nii.gz')

def path_to_subj(path):
    path = str(path)
    m = re.search(r"(IXI\d{3}-HH-\d{4})", path)
    return m.group(1)

       
class NiftiDataset(Dataset):
    """
    Extract images from the source (t1) and target (t2) directories
     
    Args:
        source: path to load source from
        target: path to load target from 
        transforms: trransforms to apply to both
    """
    def __init__(self, source_dir, target_dir, transform=None):
        
        # look in the directory and collect pairs of items. 
        t1_files = os.listdir(source_dir)
        t2_files = os.listdir(target_dir)

        self.subj_list = [ path_to_subj(e) for e in t1_files if fnmatch(e, r'*T1_fcm.nii.gz') ]        
        self.transform = transform
        
    def __len__(self):
        return len(self.subj_list)

    def __getitem__(self, idx):

        t1 = load_mr_image(self.subj_list[idx], "t1")
        t2 = load_mr_image(self.subj_list[idx], "t2")
        # TODO: should we automatically tranform np arrays to tensors? 
        sample = {'source': t1, 'target': t2}
        
        #could I run the transform on each individually?  Do I need my transform function to handle both?
        #technically the requirements call for a transform that works on a pair of images. 
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def byName(self, img_name):
        
        if re.search(r'nii.gz$', img_name):
            subj = path_to_subj(img_name)
        else:
            subj = img_name
            
        idx  = self.subj_list.index(subj)
        return self[idx]

#NOTE: there are better ways to handle this. See Subset and Sampler in Pytorch. 
class NiftiSplitDataset(NiftiDataset):
    
    def __init__(self, source_dir, target_dir, mask: List[int], transform=None, mult=1):
        super(NiftiSplitDataset, self).__init__(source_dir, target_dir, transform)
        assert(len(self.subj_list) >= len(mask))
        self.mask_subjs = mask
        self.mult = mult
    
    def __len__(self):
        return self.mult * len(self.mask_subjs)

    def __getitem__(self, idx):
        idx = idx % len(self.mask_subjs) #in case mult argument is used. TODO: this could be cleaned up to throw an error when appropriate. 
        idx = self.mask_subjs[idx]
        return super(NiftiSplitDataset, self).__getitem__(idx)
        
#transform
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
class RandomCrop3D:
    """Randomly crop randomly a 3D patch from images in an image pair.

    Args:
        output_size (tuple or int): Desired output size. If int, cube patch
            is made.
    """ 

    
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size)==3
            self.output_size = output_size

    #TODO: this function is now so small it's probably not worth keeping. 
    @staticmethod
    def crop(image, output_size, corners):
        
         #TODO: did I get the indexes right here? Might be L x H X W
        new_h, new_w, new_l = output_size
        top, left, front    = corners     
        
        # https://nipy.org/nibabel/reference/nibabel.spatialimages.html
        image = image.slicer[top:top+new_h, left:left+new_w, front:front+new_l]
        return image       
            
    def __call__(self, sample):
        
        assert sample['source'].shape[:3] == sample['target'].shape[:3]
        
        h,w,l = sample['source'].shape[:3]  
        new_h, new_w, new_l = self.output_size

        assert new_h < h and new_w < w and new_l < l
        
        top   = np.random.randint(0, h - new_h)
        left  = np.random.randint(0, w - new_w)
        front = np.random.randint(0, l - new_l)
        
        transformed = {}
        for i in ['source', 'target']:
            img_obj = sample[i]
            transformed[i] = RandomCrop3D.crop(img_obj, self.output_size, (top, left, front))
            
        return transformed

class ToNumpy:
    def __call__(self, sample):
        
        if isinstance(sample,tuple):
            s,t = sample
            return (s.get_fdata().astype('float'), t.get_fdata().astype('float'))
        
        return {k:v.get_fdata().astype('float') for k,v in sample.items() }

class AddDim:

    def __call__(self, sample):
        
        if isinstance(sample,tuple):
            s,t = sample
            return (np.expand_dims(s, axis=0), np.expand_dims(t, axis=0))
        
        return {k:np.expand_dims(v, axis=0) for k,v in sample.items() }

class ToTensor:

    def __call__(self, sample):

        was_tuple = False
        print('in ToTensor')
        if type(sample) is tuple:
            was_tuple = True
            src, trg = sample
            print('here')
            sample = { 'source': src, 'target': trg }
        
        t_dict = {}
        for i in ['source', 'target']:
            #nifti  = sample[i]
            #np_img = nifti.get_fdata()
            np_img = sample[i]
            if np_img.ndim == 3:
               tensor = torch.from_numpy(np_img.transpose(2,0,1))
            elif np_img.ndim == 4:
               tensor = torch.from_numpy(np_img.transpose(0,3,1,2)) 
            else:
               print("Unexpected number of dimensions. Skipping transpose.")
               tensor = torch.from_numpy(np_img)
            t_dict[i] = tensor.float()
            
        if was_tuple:
            return (t_dict['source'], t_dict['target'])
        
        return t_dict

class MRConvNet(nn.Module):
    def __init__(self, nChans=[16,1], kernel_size=3):
        super(MRConvNet, self).__init__()
        self.conv1 = nn.Conv3d(1, nChans[0], kernel_size, padding=1)
        self.bnorm = nn.BatchNorm3d(nChans[0])
        self.drop1 = nn.Dropout3d(p=0.3)
        self.conv2 = nn.Conv3d(nChans[0], nChans[1], kernel_size, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bnorm(x) #some debate about whether or not to apply batchnorm before or after the nonlinearity. 
        x = self.drop1(x)
        x = F.relu(self.conv2(x))       
        return x

