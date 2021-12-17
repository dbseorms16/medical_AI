import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import SimpleITK as sitk


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

        hr = self.get_patch(hr)
        hr = common.set_channel(hr, n_channels=self.args.n_colors)
        
        hr_tensor = common.np2Tensor(
            hr, rgb_range=self.args.rgb_range
        )

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
            glob.glob(os.path.join(self.dir_hr, '*/T1wCE/'))
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
        filename, _ = os.path.splitext(os.path.basename(f_hr))

        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(f_hr)
        reader.SetFileNames(dicom_files)
        input_volume = reader.Execute()

        original_spacing = input_volume.GetSpacing()
        original_size = input_volume.GetSize()

        # out_spacing = [1, 1, 1]
        # out_size = [
        #     int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        #     int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        #     int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        # resample = sitk.ResampleImageFilter()
        # resample.SetOutputSpacing(out_spacing)
        # resample.SetSize(out_size)
        # resample.SetOutputDirection(input_volume.GetDirection())
        # resample.SetOutputOrigin(input_volume.GetOrigin())
        # resample.SetTransform(sitk.Transform())
        # resample.SetDefaultPixelValue(input_volume.GetPixelIDValue())

        # input_volume = resample.Execute(input_volume)

        image_array = sitk.GetArrayFromImage(input_volume)
        sitk.WriteImage(image_array, 'image_slice.png')

        # image_slices_array = image_array[int(np.shape(image_array)[0]):int(np.shape(image_array)[0]),:,:]
        # print(image_array)
        print(image_array.shape)
        # print(image_slices_array)
        return image_array, filename

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

