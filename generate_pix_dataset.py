import random
from os import mkdir
from os.path import basename, exists, join

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm as tqdm
import torchvision.transforms as transforms

from dcgan_models import Encoder, Generator
from utils import get_jpg_images, load_encoder, load_generator, denorm


def generate_pix2pix_dataset(generator_params, encoder_params, input_dataset_path='shoes_images', dcgan_image_size=64, pix2pix_image_size=128, output_path='pix2pix/datasets/details_dataset'):
    phases = ['train','test']

    out_A_path = join(output_path,'A')
    out_B_path = join(output_path,'B')
    out_AB_path = join(output_path,'AB')

    for path in (out_A_path, out_B_path, out_AB_path):
        if not exists(path):
            mkdir(path)
        for phase in phases:
            if not exists(join(path,phase)):
                mkdir(join(path,phase))

    # useful transforms
    transform = transforms.Compose([transforms.Resize((dcgan_image_size,dcgan_image_size), Image.LANCZOS),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    interpolate = lambda x: F.interpolate(x, scale_factor=pix2pix_image_size/dcgan_image_size, mode='bilinear')

    # Load DCGAN models:
    G = load_generator(**generator_params)
    E = load_encoder(**encoder_params)

    # Iterate on images
    images_list = get_jpg_images(input_dataset_path)
    random.Random(5).shuffle(images_list)  # shuffle dataset with a constant seed (5) for consistency
    phase_cutoffs = [0.95*len(images_list), len(images_list)]
    cur_phase = 0
    for i, image_file in tqdm(enumerate(images_list)):
        if i > phase_cutoffs[cur_phase]:
            cur_phase += 1
        with Image.open(image_file) as image:
            in_image = transform(image.convert("RGB")).cuda()
            if tuple(in_image.shape[-3:]) != (3,dcgan_image_size,dcgan_image_size):
                print(f"WARNING! Unexpected input size: {in_image.shape} in file {image_file}. Skipping...")
                continue
            B_image = image.resize((pix2pix_image_size,pix2pix_image_size), Image.BILINEAR)
            B_image.save(join(out_B_path,phases[cur_phase],basename(image_file)[:-3]+"png"))
        generated_image = G(E(in_image.reshape(1,3,dcgan_image_size,dcgan_image_size)))
        upsampled = interpolate(generated_image)
        fixed_point = np.uint8(np.round(255*denorm(upsampled).cpu().numpy()))[0,...]
        fixed_point = np.transpose(fixed_point, (1,2,0))
        A_image = Image.fromarray(fixed_point)
        A_image.save(out_A_path+'/'+phases[cur_phase]+'/'+basename(image_file)[:-3]+"png")
        
        w, h = A_image.size
        AB_image = Image.new("RGB", (2*w, h))
        AB_image.paste(A_image, (0,0))
        AB_image.paste(B_image, (w,0))
        AB_image.save(join(out_AB_path,phases[cur_phase],basename(image_file)[:-3]+"png"))
