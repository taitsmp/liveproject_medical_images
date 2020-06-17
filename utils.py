
from torch.utils.data.dataset import Dataset
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

class NiftiSplitDataset(NiftiDataset):
    
    def __init__(self, source_dir, target_dir, mask: List[int], transform=None):
        super(NiftiSplitDataset, self).__init__(source_dir, target_dir, transform)
        assert(len(self.subj_list) >= len(mask))
        self.mask_subjs = mask
    
    def __len__(self):
        return len(self.mask_subjs)

    def __getitem__(self, idx):
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
        

