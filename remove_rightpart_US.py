import os
#from polyaxon_helper import get_data_paths
root = "/mnt/data/prostate_data/preprocess_3/projects/DeepProstateDB/Data"

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
            mhd_old = open(subdir + "/trus.mhd", "r")
            for line in mhd_old:
                if "Orientation" in line:
                    orientation = line

            image = sitk.ReadImage(subdir + "/trus.mhd")


            img = sitk.GetArrayFromImage(image)
            #fig = plt.figure()
            #ax1 = fig.add_subplot(2, 2, 1)
            #ax1.imshow(img[:, :, 40])
            for depth in range(img.shape[2]):
                for row in range(img.shape[0]):
                    for pixel in reversed(range(img.shape[1])):
                        if img[row, pixel,depth] > 0.0:
                            img[row, pixel-4:pixel+1, depth] = 0.0
                            break

            dir = subdir + "/trus_cut.mhd"
          #  ax2 = fig.add_subplot(2, 2, 2)
          #  ax2.imshow(img[:,:,40])
          #  plt.show()
            new_itk_image = sitk.GetImageFromArray(img)
            new_itk_image.SetOrigin(image.GetOrigin())
            new_itk_image.SetSpacing(image.GetSpacing())
            new_itk_image.SetDirection(image.GetDirection())
            sitk.WriteImage(new_itk_image, dir)
            mhd_new = open(dir, "a")
            mhd_new.write(orientation)
            mhd_new.close()