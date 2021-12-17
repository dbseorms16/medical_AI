import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import SimpleITK as sitk
import torch

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()
    
    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)

        # hr = self.get_patch(hr)
        # hr = common.set_channel(hr, n_channels=self.args.n_colors)
        hr_tensor = torch.from_numpy(hr).float()
        
        # hr_tensor = common.np2Tensor(
        #     hr, rgb_range=self.args.rgb_range
        # )

        return hr_tensor, filename

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_hr = self._scan()
        self.images_hr = list_hr

    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = self.dataset_length // len(self.images_hr)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*/T1wCE'))
        )
        return names_hr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.ext = ('.png', '.png')

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        hr_imges = sorted(glob.glob(f_hr+'/*.png'), key=lambda x : int(x.split('-')[1].split('.')[0]))

        mid_i = len(hr_imges) // 2 - 20

        hr = [imageio.imread(hr_imges[i * 10 - mid_i]) for i in range(5)]
        hr = np.array(hr)
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        return hr, filename

    def get_patch(self, hr):
        scale = self.scale
        # multi_scale = len(self.scale) > 1
        # if self.train:
        #     if not self.args.no_augment:
        #         lr, hr = common.augment(lr, hr)
        # else:
        #     if isinstance(lr, list):
        #         ih, iw = lr[0].shape[:2]
        #     else:
        #         ih, iw = lr.shape[:2]
        #     hr = hr[0:ih * scale[0], 0:iw * scale[0]]
            
        return hr

