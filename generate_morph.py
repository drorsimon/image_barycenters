import sys
sys.path.insert(0,"pix2pix")  # fix pix2pix model's path

import argparse
import os
from os import listdir
from os.path import basename, isfile, join
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from numba import jit
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

import utils
from pix2pix.models import networks
from utils import get_jpg_images
from dcgan_models import Generator, Encoder


parser = argparse.ArgumentParser()
parser.add_argument("--dcgan_latent_size", type=int, dest="dcgan_latent_size", help="The size of the latent vectors used in the DCGAN model", default=100)
parser.add_argument("--dcgan_num_filters", type=List[int], dest="dcgan_num_filters", help="Num of filters in each of the DCGAN's layers", default=[1024, 512, 256, 128])
parser.add_argument("--dcgan_image_size", type=int, dest="dcgan_image_size", help="input/output size for the DCGAN model", default=64)
parser.add_argument("--models_save_dir", type=str, dest="models_save_dir", help="The path to load the trained models from", default="networks")
parser.add_argument("--dataset_name", type=str, dest="dataset_name", help="Name of the used dataset", default="zap50k")
parser.add_argument("--dataset_base_dir", type=str, dest="dataset_base_dir", help="Path to the base dir of the dataset's jpg images", default="dataset/ut-zap50k-images-square")
parser.add_argument("--pix2pix_image_size", type=int, dest="pix2pix_image_size", help="input/output size for the pix2pix model. 128|256", default=128)
parser.add_argument("--results_folder", type=str, dest="results_folder", help="Path to store the interpolation results", default="results")
parser.add_argument("--entropy_regularization", type=float, dest="entropy_regularization", help="Value for the Wasserstein distance's entropy regularization", default=20.0)
parser.add_argument("--interpolation_steps", type=float, dest="interpolation_steps", help="Number of interpolation steps in the trainsformation", default=9)
parser.add_argument("--simulation_name", type=str, dest="simulation_name", help="A name for the simulation out file. If not set, simulation will be named by the chosen file names", default=None)
parser.add_argument("--image_1_path", type=str, dest="image_1_path", help="A path to the first image to interpolate from. If none, an image is selected randomly from the dataset.", default=None)
parser.add_argument("--image_2_path", type=str, dest="image_2_path", help="A path to the second image to interpolate to. If none, an image is selected randomly from the dataset.", default=None)
args = parser.parse_args()

# Compute the L2 metric for the transportation cost. Can probably be vectorized to run faster.
@jit("float64[:,:](int64,int64,int64[:,:,:])",nopython=True)
def _generate_metric(height, width, grid):
    # Could probably inpmprove runtime using vectorized code
    C = np.zeros((height*width, height*width))
    i = 0
    j = 0
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    C[i,j] = np.square(grid[x1,y1,:] - grid[x2,y2,:]).sum()
                    j += 1
            j = 0
            i += 1
    return C

def generate_metric(im_size: Tuple[int]) -> np.ndarray:
    """
    Computes the Euclidean distances matrix
    
    Arguments:
        im_size {Tuple[int]} -- Size of the input image (height, width)
    
    Returns:
        np.ndarray -- distances matrix
    """
    grid = np.meshgrid(*[range(x) for x in im_size])
    grid = np.stack(grid,-1)
    return _generate_metric(im_size[0], im_size[1], grid)

# Find interpolation given the transportation plan. Can probably be vectorized to run faster.
@jit("float64[:,:](int64,int64,float64[:,:,:,:],float32)",nopython=True)
def generate_interpolation(height, width, plan, t):
    c = np.zeros((height+1, width+1))
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    new_loc_x = (1-t)*x1 + t*x2
                    new_loc_y = (1-t)*y1 + t*y2
                    p = new_loc_x - int(new_loc_x)
                    q = new_loc_y - int(new_loc_y)
                    c[int(new_loc_x),int(new_loc_y)] += (1-p)*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)] += p*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x),int(new_loc_y)+1] += (1-p)*q*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)+1] += p*q*plan[x1,y1,x2,y2]
    c = c[:height,:width] #* (I1_count*(1-t) + I2_count*t)
    return c

def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, height: int, width: int, 
             epsilon: float, threshold: float=1e-7) -> np.ndarray:
    """Computes the sinkhorn algorithm naively, using the CPU.
    
    Arguments:
        a {np.ndarray} -- the first distribution (image), normalized, and shaped to a vector of size height*width.
        b {np.ndarray} -- the second distribution (image), normalized, and shaped to a vector of size height*width.
        C {np.ndarray} -- the distances matrix
        height {int} -- image height
        width {int} -- image width
        epsilon {float} -- entropic regularization parameter
    
    Keyword Arguments:
        threshold {float} -- convergence threshold  (default: {1e-7})
    
    Returns:
        np.ndarray -- the entropic regularized transportation plan, pushing distribution a to b.
    """
    K = np.exp(-C/epsilon)
    v = np.random.randn(*a.shape)
    i = 0
    while True:
        u = a/(K.dot(v))
        v = b/(K.T.dot(u))
        i += 1
        if i % 50 == 0:
            convergence = np.square(np.sum(u.reshape(-1, 1) * K * v.reshape(1,-1), axis=1) - a).sum()
            if convergence < threshold:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} (Converged!)")
                break
            else:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} ( > {threshold})")

    P = u.reshape(-1, 1) * K * v.reshape(1,-1)
    P = P.reshape(height, width, height, width)
    return P

def sinkhorn_gpu(a: np.ndarray, b: np.ndarray, C: np.ndarray, height: int, width: int, 
                 epsilon: float, threshold: float=1e-7) -> np.ndarray:
    """Computes the sinkhorn algorithm using convolutional Wassestein distances (Solomon et al.), using the GPU.
    
    Arguments:
        a {np.ndarray} -- the first distribution (image), normalized, and shaped to a vector of size height*width.
        b {np.ndarray} -- the second distribution (image), normalized, and shaped to a vector of size height*width.
        C {np.ndarray} -- the distances matrix
        height {int} -- image height
        width {int} -- image width
        epsilon {float} -- entropic regularization parameter
    
    Keyword Arguments:
        threshold {float} -- convergence threshold  (default: {1e-7})
    
    Returns:
        np.ndarray -- the entropic regularized transportation plan, pushing distribution a to b.
    """
    K = torch.tensor(np.exp(-C/epsilon), dtype=torch.double).cuda()

    # Compute the kernel:
    kernel_size = max(int(epsilon*2),7)  # seems to be sufficient
    if kernel_size % 2 == 0:
        kernel_size += 1
    X = torch.tensor(range(kernel_size)).repeat(kernel_size,1)
    Y = torch.tensor(range(kernel_size)).reshape(-1,1).repeat(1,kernel_size)
    D = torch.zeros_like(X)
    for i in range(kernel_size):
        for j in range(kernel_size):
            D[i,j] = (X[i,j] - X[kernel_size//2,kernel_size//2])**2 + (Y[i,j] - Y[kernel_size//2,kernel_size//2])**2
    k = torch.exp(-D.type(torch.double)/epsilon).cuda().reshape(1,1,kernel_size,kernel_size)
    
    a = torch.tensor(a.reshape(1,1,height,width), dtype=torch.double).cuda()
    a /= a.sum()
    b = torch.tensor(b.reshape(1,1,height,width), dtype=torch.double).cuda()
    b /= b.sum()
    v = torch.ones_like(a) / k.sum()
    i = 0
    a_mask = a==0
    b_mask = b==0
    last_convergence = np.inf

    # Compute convolutional Sinkhorn
    while True:
        u = a/(torch.nn.functional.conv2d(v, k, padding=kernel_size//2))
        u[a_mask] = 0  # Fix 0/0 devision
        v = b/(torch.nn.functional.conv2d(u, k, padding=kernel_size//2))
        v[b_mask] = 0  # Fix 0/0 devision
        i += 1
        if i % 10 == 0:
            convergence = (torch.sum(u.reshape(-1, 1) * K * v.reshape(1,-1), dim=1) - a.reshape(-1)).pow(2).sum().item()  # Could probably be computed using convs as well
            print(i, convergence)
            if convergence < threshold or np.abs(last_convergence - convergence) < threshold:
                break
            last_convergence = convergence

    # Compute the trasportation plan
    P = u.reshape(-1, 1) * K * v.reshape(1,-1)  # Could probably be computed using convs as well
    P = P.reshape(height, width, height, width).cpu().numpy().astype(np.float)
    return P

def project_on_generator(G: Generator, pix2pix: networks.UnetGenerator, 
                         target_image: np.ndarray, E: Encoder, dcgan_img_size: int=64, 
                         pix2pix_img_size: int=128) -> Tuple[np.ndarray, torch.Tensor]:
    """Projects the input image onto the manifold span by the GAN. It operates as follows:
    1. reshape and normalize the image
    2. run the encoder to obtain a latent vector
    3. run the DCGAN generator to obtain a low resolution image
    4. run the Pix2Pix model to obtain a high resulution image
    
    Arguments:
        G {Generator} -- DCGAN generator
        pix2pix {networks.UnetGenerator} -- Low resolution to high resolution Pix2Pix model
        target_image {np.ndarray} -- The image to project
        E {Encoder} -- The DCGAN encoder
    
    Keyword Arguments:
        dcgan_img_size {int} -- Low resolution image size (default: {64})
        pix2pix_img_size {int} -- High resolution image size (default: {128})
    
    Returns:
        Tuple[np.ndarray, torch.Tensor] -- The projected high resolution image and the latent vector that was used to generate it.
    """
    # reshape and normalize image
    target_image = torch.Tensor(target_image).cuda().reshape(1,3,pix2pix_img_size,pix2pix_img_size)
    target_image = F.interpolate(target_image, scale_factor=dcgan_img_size/pix2pix_img_size, mode='bilinear')
    target_image = target_image.clamp(min=0)
    target_image = target_image / target_image.max()
    target_image = (target_image - 0.5) / 0.5

    # Run dcgan
    z = E(target_image)
    dcgan_image = G(z)

    # run pix2pix
    pix_input = F.interpolate(dcgan_image, scale_factor=pix2pix_img_size/dcgan_img_size, mode='bilinear')
    pix_outputs = pix2pix(pix_input)
    out_image = utils.denorm(pix_outputs.detach()).clamp(0,1).cpu().numpy().reshape(3,-1,1)
    return out_image, z

def preprocess_Q(Q: np.ndarray, max_val: float=None, Q_counts: np.ndarray=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """ Preprocess (normalize) input images before computing their barycenters
    
    Arguments:
        Q {np.ndarray} -- Input images. Every image should reshaped to a column in Q.
    
    Keyword Arguments:
        max_val {float} -- The maximum value. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
        Q_counts {np.ndarray} -- The sum of all the pixel values in each image. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
    
    Returns:
        Tuple[np.ndarray, float, np.ndarray] -- The normalized images the total maximum value and sum of pixels in each image
    """
    if max_val is None:
        max_val = Q.max()
    Q = max_val - Q
    if Q_counts is None:
        Q_counts = np.sum(Q, axis=1, keepdims=True)
    Q = Q / Q_counts
    return Q, max_val, Q_counts

def load_pix2pix(model_path: str='networks/zap50k_pix2pix') -> networks.UnetGenerator:
    """Loads the Pix2Pix model
    
    Keyword Arguments:
        model_path {str} -- Path to the trained Pix2Pix model (default: {'networks/zap50k_pix2pix'})
    
    Returns:
        networks.UnetGenerator -- An object that holds the Pix2Pix conditional generator model
    """
    netG = networks.define_G(3, 3, 64, 'unet_128', 'batch', True, 'normal', 0.02, [0]).module
    netG.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False
    return netG

def load_models() -> Tuple[Generator, Encoder, networks.UnetGenerator]:
    """ Load the generative models
    
    Returns:
        Tuple[Generator, Encoder, networks.UnetGenerator] -- the DCGAN generator, its respective encoder and the Pix2Pix model
    """
    generator = utils.load_generator(args.dcgan_latent_size, args.dcgan_num_filters, join(args.models_save_dir,'generator'))
    encoder = utils.load_encoder(args.dcgan_latent_size, args.dcgan_num_filters, join(args.models_save_dir,'encoder'))
    pix2pix = load_pix2pix(join(args.models_save_dir, args.dataset_name+"_pix2pix"))
    return generator, encoder, pix2pix

def morph_project_only(im1: np.ndarray, im2: np.ndarray, Generator: Generator, Encoder: Encoder, 
                       pix2pix: networks.UnetGenerator, epsilon: float=20.0, L: int=9, dcgan_size: int=64, 
                       pix2pix_size: int=128, simulation_name: str="image_interpolation", results_path: str="results") -> None:
    """Generates 3 morphing processes given two images. 
    The first is simple Wasserstein Barycenters, the second is our algorithm and 
    the third is a simple GAN latent space linear interpolation
    
    Arguments:
        im1 {np.ndarray} -- source image
        im2 {np.ndarray} -- destination image
        Generator {Generator} -- DCGAN generator (latent space to pixel space)
        Encoder {Encoder} -- DCGAN encoder (pixel space to latent space)
        pix2pix {networks.UnetGenerator} -- pix2pix model trained to increase an image resolution
    
    Keyword Arguments:
        epsilon {float} -- entropic regularization parameter (default: {20.0})
        L {int} -- number of images in the trasformation (default: {9})
        dcgan_size {int} -- DCGAN image size (low resolution) (default: {64})
        pix2pix_size {int} -- Pix2Pix image size (high resolution) (default: {128})
        simulation_name {str} -- name of the simulation. Affects the saved file names (default: {"image_interpolation"})
        results_path {str} -- the path to save the results in (default: {"results"})
    """
    img_size = im1.shape[:2]
    im1, im2 = (I.transpose(2,0,1).reshape(3,-1,1) for I in (im1, im2))

    print("Preparing transportation cost matrix...")
    C = generate_metric(img_size)
    Q = np.concatenate([im1, im2], axis=-1)
    Q, max_val, Q_counts = preprocess_Q(Q)
    out_ours = []
    out_GAN = []
    out_OT = []

    print("Computing transportation plan...")
    for dim in range(3):
        print(f"Color space {dim+1}/3")
        out_OT.append([])
        P = sinkhorn(Q[dim,:,0], Q[dim,:,1], C, img_size[0], img_size[1], epsilon)
        for t in tqdm(np.linspace(0,1,L)):
            out_OT[-1].append(max_val - generate_interpolation(img_size[0],img_size[1],P,t)*((1-t)*Q_counts[dim,0,0] + t*Q_counts[dim,0,1]))
    out_OT = [np.stack(im_channels, axis=0) for im_channels in zip(*out_OT)]
    
    print("Computing GAN projections...")
    # Project OT results on GAN
    GAN_projections = [project_on_generator(Generator, pix2pix, I, Encoder, dcgan_img_size=dcgan_size, pix2pix_img_size=pix2pix_size) for I in out_OT]
    GAN_projections_images, GAN_projections_noises = zip(*GAN_projections)
    out_ours = GAN_projections_images

    # Linearly interpolate GAN's latent space
    noise1, noise2 = GAN_projections_noises[0].cuda(), GAN_projections_noises[-1].cuda()
    for t in np.linspace(0,1,L):
        t = float(t)  # cast numpy object to primative type
        GAN_image = Generator((1-t)*noise1 + t*noise2)
        GAN_image = F.interpolate(GAN_image, scale_factor=2, mode='bilinear')
        pix_outputs = pix2pix(GAN_image)
        GAN_image = utils.denorm(pix_outputs.detach()).cpu().numpy().reshape(3,-1,1)
        out_GAN.append(GAN_image.clip(0,1))
            
    # Save results:
    print("Saving results...")
    out_ours = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_ours])
    out_OT = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_OT])
    out_GAN = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_GAN])
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    output_path = join(results_path, simulation_name+'.png')
    save_image(torch.cat([out_OT,out_ours,out_GAN], dim=0), output_path, nrow=L, normalize=False, scale_each=False, range=(0,1))
    print(f"Image saved in {output_path}")

def load_images(path1: str, path2: str, im_size: int=128) -> Tuple[np.ndarray, np.ndarray, str]:
    """Loads images
    
    Arguments:
        path1 {str} -- path to the first image
        path2 {str} -- path to the second image
    
    Keyword Arguments:
        im_size {int} -- the desired image size (default: {128})
    
    Returns:
        Tuple[np.ndarray, np.ndarray, str] -- returns the two images and a the files' names (used later to save to results)
    """
    im1, im2 = (np.array(Image.open(path).convert("RGB").resize((im_size,im_size),Image.LANCZOS), dtype=np.float)/255 for path in (path1, path2))
    files_basename = '_'.join([basename(path).split('.')[0] for path in (path1, path2)])
    return im1, im2, files_basename


if __name__=='__main__':
    # Load images
    print("Loading images...")
    if args.image_1_path is None or args.image_2_path is None:
        image_list = get_jpg_images(args.dataset_base_dir)
        images_paths = np.random.choice(image_list,size=2,replace=False)
    images_paths = [x if x is not None else images_paths[i] for i,x in enumerate((args.image_1_path,args.image_2_path))]
    im1, im2, files_basename = load_images(*images_paths, im_size=args.pix2pix_image_size)
    
    # Load models
    print("Loading models...")
    generator, encoder, pix2pix = load_models()
    simulation_name = args.simulation_name if args.simulation_name is not None else files_basename

    print("Interpolating images:")
    morph_project_only(im1, im2, generator, encoder, pix2pix, args.entropy_regularization, args.interpolation_steps, args.dcgan_image_size, args.pix2pix_image_size, simulation_name, args.results_folder)
