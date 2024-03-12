import numpy as np
import pydicom
from PIL import Image
import glob
import os
import pathlib
# import system

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_path = "mammo-analysis/data/data_raw/"
data_processed = 'mammo-analysis/data/data_processed/'

def convert_jpg():

    # Grabing filenames.
    dicom_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))


    # Converting files
    for i in dicom_files:
        img = pydicom.dcmread(i)

        img = img.pixel_array.astype(float)

        rescaled_image = (np.maximum(img,0)/img.max())*255
        final_image = np.uint8(rescaled_image)

        final_image = Image.fromarray(final_image)

        final_image.save(f"{data_processed}{os.path.split(i)[-1]}.jpg")


    # Removing .dcm:
    paths = (os.path.join(root, filename)
            for root, _, filenames in os.walk(data_processed)
            for filename in filenames)

    for path in paths:
        newname = path.replace('.dcm', '')
        if newname != path:
            os.rename(path, newname)


convert_jpg()
