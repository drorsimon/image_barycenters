import argparse
from typing import List
from generate_h5 import generate_h5
import train_dcgan
import os
from os.path import join
import torch
from generate_pix_dataset import generate_pix2pix_dataset
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, dest="dataset_name", help="Name of the used dataset", default="zap50k")
parser.add_argument("--dataset_base_dir", type=str, dest="dataset_base_dir", help="Path to the base dir of the dataset's jpg images", default="dataset/ut-zap50k-images-square")
parser.add_argument("--dataset_h5_save_dir", type=str, dest="dataset_h5_save_dir", help="Path to save the preprocessed dataset images", default="dataset/ut-zap50k-images-square")
parser.add_argument("--dcgan_image_size", type=int, dest="dcgan_image_size", help="input/output size for the DCGAN model", default=64)
parser.add_argument("--dcgan_latent_size", type=int, dest="dcgan_latent_size", help="The size of the latent vectors used in the DCGAN model", default=100)
parser.add_argument("--dcgan_num_filters", type=List[int], dest="dcgan_num_filters", help="Num of filters in each of the DCGAN's layers", default=[1024, 512, 256, 128])
parser.add_argument("--dcgan_batch_size", type=int, dest="dcgan_batch_size", help="The batch size to use in the DCGAN training phase", default=128)
parser.add_argument("--dcgan_num_epochs", type=int, dest="dcgan_num_epochs", help="Num of DCGAN training epochs", default=100)
parser.add_argument("--dcgan_log_dir", type=str, dest="dcgan_log_dir", help="DCGAN training phase logdir (for figures and images)", default='dcgan_log_dir')
parser.add_argument("--models_save_dir", type=str, dest="models_save_dir", help="The path to save the trained models at", default="networks")
parser.add_argument("--pix2pix_dataset_name", type=str, dest="pix2pix_dataset_name", help="The given name to the dataset used to train the pix2pix model.", default="details_dataset")
parser.add_argument("--pix2pix_image_size", type=int, dest="pix2pix_image_size", help="input/output size for the pix2pix model. 128|256", default=128)
parser.add_argument("--pix2pix_batch_size", type=int, dest="pix2pix_batch_size", help="The batch size to use in the pix2pix training phase", default=128)



args = parser.parse_args()


def preprocess_dataset():
    # Create dataset to train the DCGAN
    h5_path = join(args.dataset_h5_save_dir, args.dataset_name+'.hdf5')
    if not os.path.exists(h5_path):
        generate_h5(args.dataset_base_dir, args.dataset_h5_save_dir, args.dcgan_image_size, args.dataset_name)

def train_dcgan(test_encoder=True):
    dcgan_args = {'latent_dim':args.dcgan_latent_size, 
                'num_filters':args.dcgan_num_filters, 
                'batch_size':args.dcgan_batch_size,
                'num_epochs':args.dcgan_num_epochs, 
                'h5_file_path':join(args.dataset_h5_save_dir, args.dataset_name+'.hdf5'), 
                'save_dir':args.models_save_dir, 
                'train_log_dir':args.dcgan_log_dir}

    if not os.path.exists(args.models_save_dir):
        os.mkdir(args.models_save_dir)
 
    train_dcgan.train_gan(**dcgan_args)
    train_dcgan.train_encoder_with_noise(**dcgan_args)
    train_dcgan.finetune_encoder_with_samples(**dcgan_args)
    if test_encoder:
        train_dcgan.test_encoder(**dcgan_args)

def prepare_pix2pix_dataset():
    pix2pix_dataset_path = join("pix2pix/datasets/", args.pix2pix_dataset_name)
    if not os.path.exists(pix2pix_dataset_path):
        os.mkdir(pix2pix_dataset_path)

    generator_params = {'latent_size':args.dcgan_latent_size, 'num_filters':args.dcgan_num_filters, 'generator_path':join(args.models_save_dir,'generator')}
    encoder_params = {'latent_size':args.dcgan_latent_size, 'num_filters':args.dcgan_num_filters, 'encoder_path':join(args.models_save_dir,'encoder')}
    generate_pix2pix_dataset(generator_params, encoder_params, args.dataset_base_dir, args.dcgan_image_size, args.pix2pix_image_size, pix2pix_dataset_path)

def train_pix2pix():
    os.chdir("pix2pix")
    pix2pix_dataset_path = join("datasets/", args.pix2pix_dataset_name)
    pix2pix_params = ' '.join(["--dataroot", join(pix2pix_dataset_path,'AB'),
                               "--name", args.dataset_name+"_pix2pix",
                               "--model", "pix2pix",
                               "--netG", "unet_"+str(args.pix2pix_image_size),
                               "--direction", "AtoB",
                               "--lambda_L1", "100",
                               "--dataset_mode", "aligned",
                               "--norm", "batch",
                               "--pool_size", "0",
                               "--no_flip",
                               "--preprocess", "none",
                               "--batch_size", str(args.pix2pix_batch_size)])
    os.system("python train.py " + pix2pix_params)
    os.chdir("..")

    copyfile(join("pix2pix/checkpoints", args.dataset_name+"_pix2pix", "latest_net_G.pth"), join(args.models_save_dir,args.dataset_name+"_pix2pix"))

if __name__ == '__main__':
    # preprocess_dataset()
    # train_dcgan()
    # prepare_pix2pix_dataset()
    train_pix2pix()