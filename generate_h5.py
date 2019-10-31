from os import listdir, mkdir, path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from utils import get_jpg_images


def generate_h5(base_dir='shoes_images', save_dir='shoes_images', image_size=64, dataset_name='shoes'):
    transform = transforms.Compose([transforms.Resize((image_size,image_size), Image.LANCZOS),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    if not path.exists(save_dir):
        mkdir(save_dir)
    h5_file = h5py.File(path.join(save_dir,dataset_name+'.hdf5'),'w')

    images_list = get_jpg_images(base_dir)
    N = len(images_list)
    data = h5_file.create_dataset('data', shape=(N, 3, image_size, image_size), dtype=np.float32, fillvalue=0)
    for i, image_file in tqdm(enumerate(images_list)):
        with Image.open(image_file) as image:
            out_image = transform(image)
            try:
                data[i] = out_image
            except:
                print(image_file, out_image.shape)
                break

    h5_file.close()


if __name__ == '__main__':
    generate_h5()
