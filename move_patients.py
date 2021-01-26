import shutil
import os

from polyaxon_helper import get_data_paths

## For NAS
base_path = get_data_paths()
source_dir = base_path['data1'] + "/Prostate_Reg_KM/anonymized"
destination_dir= base_path['data1'] + "/Prostate_Reg_KM/shi_images"

f = open("patients_tomove.txt", "r")
content_list = [line.rstrip('\n') for line in f]
f.close()

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for subdir, dirs, files in os.walk(source_dir):
    print(os.path.basename(subdir).strip())
    if os.path.basename(subdir).strip() in content_list:
        shutil.move(subdir,os.path.join(destination_dir,os.path.basename(subdir)))