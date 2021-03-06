import os
#from polyaxon_helper import get_data_paths
root = "/mnt/data/prostate_data/preprocess_5/projects/DeepProstateDB/Data/Anonymized_cleaned/train"

import SimpleITK as sitk
import matplotlib.pyplot as plt

## For NAS
#base_path = get_data_paths()
#root = base_path['data1'] + "/Prostate_Reg_KM/anonymized"

#root = "/Users/kristinamach/Desktop/anonymized/train"

for subdir, dirs, files in os.walk(root):
    for file in files:
        orientation = ""
        if "trus.mhd" in file and "cropped" in subdir:
            image = sitk.ReadImage(subdir + "/trus.mhd")
            img = sitk.GetArrayFromImage(image)

            fig = plt.figure()

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(img[:,:,40], cmap="gray")

            plt.show()
