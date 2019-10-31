from dcgan_models import Generator, Encoder
import torch
from os import listdir, path


def get_jpg_images(p: str):
    if not path.isdir(p):
        if p.endswith('.jpg'):
            return [p]
        return None
    filelist = []
    for fname in listdir(p):
        cur_filelist = get_jpg_images(path.join(p,fname))
        if cur_filelist is not None:
            filelist.extend(cur_filelist)
    return filelist

def load_generator(latent_size=100, num_filters=[1024, 512, 256, 128], generator_path='networks/generator'):
    G = Generator(latent_size, num_filters).cuda()
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False
    return G

def load_encoder(latent_size=100, num_filters=[1024, 512, 256, 128], encoder_path='networks/encoder'):
    E = Encoder(num_filters[::-1], latent_size).cuda()
    E.load_state_dict(torch.load(encoder_path))
    E.eval()
    for param in E.parameters():
        param.requires_grad = False
    return E

def alexnet_norm(x): 
    assert x.max() <= 1 or x.min() >= 0, f"Alexnet received input outside of range [0,1]: {x.min(),x.max()}"
    out = x - torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).type_as(x)
    out = out / torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).type_as(x)
    return out

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
