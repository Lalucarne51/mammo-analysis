import numpy as np
import pydicom
import pylibjpeg
from PIL import Image
import os
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms

data_path = r"C:\Users\pier3\OneDrive\Bureau\PROJECT\Kaggle dataset\rnsa_data\train_images"
data_processed = r'C:\Users\pier3\OneDrive\Bureau\PROJECT\Kaggle dataset\rnsa_data\images_processed_2\images_prossessed'


def convert_dicom_to_jpg():

    # Grabing filenames.
    dicom_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))


    # Converting files
    for i in tqdm(dicom_files):
        img = pydicom.dcmread(i)
        need_flip = False
        if img.PhotometricInterpretation == 'MONOCHROME1':
            need_flip = True
        img.PhotometricInterpretation = 'YBR_FULL'

        img = img.pixel_array.astype(float)

        rescaled_image = (np.maximum(img,0)/img.max())*255
        if need_flip == True:
            rescaled_image = 1 - rescaled_image
        final_image = np.uint8(rescaled_image)

        final_image = Image.fromarray(final_image)

        return final_image.save(f"{data_processed}{os.path.split(i)[-1]}.jpg")



def name_cleaning():
    # Removing .dcm:
    paths = (os.path.join(root, filename)
            for root, _, filenames in os.walk(data_processed)
            for filename in filenames)

    for path in tqdm(paths):
        newname = path.replace('.dcm', '')
        newname = newname.replace('img_preprocessed', '')
        if newname != path:
            os.rename(path, newname)




def crop_and_convert_torch_tensor(BUCKET_NAME="BUCKET_NAME", IMAGE_ID="IMAGE_ID"):
    # CONVERTION EN TENSOR,
    # DEPUIS LE MEME DOSSIER !!!!!!!!!!

    # Crop and save a copy of the image
    image = Image.open(f'{BUCKET_NAME}/{IMAGE_ID}')
    cropped = ImageOps.crop(image, 256)
    cropped.save(f"image_cropped_{IMAGE_ID}")

    # Define a transform to convert PIL
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(cropped)

    # print the converted Torch tensor
    return img_tensor



def crop_and_convert_np_array():
    # CONVERTION EN NP.ARRAY,
    # DEPUIS LE MEME DOSSIER !!!!!!!!!!
    image = Image.open(image)
    cropped = ImageOps.crop(image, 256)
    cropped.save(f"{image.name}.jpg")

    np_array = np.array(cropped)

    return np_array
