import SimpleITK as sitk
import glob
import os
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
names_hr = sorted(
            glob.glob(os.path.join('./train/*/T1wCE/*.dcm')))

tqdm_test = tqdm(names_hr, ncols=80)

for hr in tqdm_test:
    img = sitk.ReadImage(hr)
    img = sitk.GetArrayFromImage(img)
    if np.max(img) != 0:
        img = img / np.max(img) *255.0
    # print(hr.split('.dcm'))
    
    imageio.imwrite('{}.png'.format(hr.split('.dcm')[0]), img.transpose(1,2,0).astype(np.uint8))