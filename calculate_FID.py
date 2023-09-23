import torch
import torch.nn.functional as F
import torch.nn
import argparse
import utils
import numpy as np
from generator import Generator
from torchvision.models import inception_v3
from scipy.linalg import sqrtm



def prepare_for_inception_v3(imgs_batch):
    """
    Prepares the given batch of images for inception_v3 model.
    
    Parameters:
        imgs_batch (torch.Tensor): A batch of images to prepare.
        
    Returns:
        torch.Tensor: The prepared batch of images.
    """
    imgs_resized = F.interpolate(imgs_batch, size=(299, 299), mode='bilinear', align_corners=False)
    imgs_normalized = (imgs_resized * 2) - 1
    return imgs_normalized


def get_real_batch(trainloader):
    """
    Gets a batch of real images from the training loader and prepares it for inception_v3 model.
    
    Parameters:
        trainloader (DataLoader): The training data loader.
        
    Returns:
        torch.Tensor: A batch of real images prepared for inception_v3 model.
    """
    data_iter = iter(trainloader)
    real_images = next(data_iter)[0]
    return prepare_for_inception_v3(real_images)


def generate_fake_batch(generator):
    """
    Generates a batch of fake images using the given generator.
    
    Parameters:
        generator (Generator): The generator model to generate fake images.
        
    Returns:
        torch.Tensor: A batch of fake images prepared for inception_v3 model.
    """
    z = torch.randn(64, 100).cuda()  # Consider using .to(device) instead of .cuda()
    fake_imgs = generator(z)
    return prepare_for_inception_v3(fake_imgs)


def calculate_fid(real_imgs, fake_imgs):
    """
    Calculates the Frechet Inception Distance (FID) between real and fake images.
    
    Parameters:
        real_imgs (torch.Tensor): A batch of real images.
        fake_imgs (torch.Tensor): A batch of fake images.
        
    Returns:
        torch.Tensor: The calculated FID.
    """
    real_imgs, fake_imgs = real_imgs.to(device), fake_imgs.to(device)
    mu_real, sigma_real = compute_statistics(real_imgs)
    mu_fake, sigma_fake = compute_statistics(fake_imgs)
    sum_sq_diff = torch.sum((mu_real - mu_fake) ** 2)
    sigma_sqrt = sqrtm((sigma_real @ sigma_fake).cpu().numpy())
    
    if np.iscomplexobj(sigma_sqrt):
        sigma_sqrt = sigma_sqrt.real
    
    fid = sum_sq_diff + torch.trace(sigma_real + sigma_fake - 2 * torch.tensor(sigma_sqrt, device=device, dtype=torch.float32))
    return fid


def compute_statistics(imgs):
    """
    Computes the statistics (mean and covariance) for the given batch of images using inception_v3 model.
    
    Parameters:
        imgs (torch.Tensor): A batch of images.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The calculated mean and covariance of the features.
    """
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    with torch.no_grad():
        features = model(imgs).view(imgs.size(0), -1)
    mu = torch.mean(features, dim=0)
    sigma = torch_cov(features, rowvar=False)
    return mu, sigma


def torch_cov(m, rowvar=False):
    """
    Computes the covariance matrix of a given matrix.
    
    Parameters:
        m (torch.Tensor): A 2D matrix.
        rowvar (bool): If True, treat the rows as variables, otherwise, treat columns as variables.
        
    Returns:
        torch.Tensor: The covariance matrix.
    """
    if m.size(0) == 1:
        return torch.empty((m.size(1), m.size(1))).fill_(0).to(device)
    if rowvar:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()
 

if __name__ == "__main__":
    
    device = torch.device("cpu")

    
    parser = argparse.ArgumentParser(description='If Inversion-Specific GAN is used')
    parser.add_argument('--improved_flag', action='store_true', default=False,
                        help='')
    args = parser.parse_args()

    file = "./config/classify.json"
    config_args = utils.load_json(json_file=file)
    train_file = config_args['dataset']['train_file_path']

    print(f"using improved model: {args.improved_flag}")
    

    if args.improved_flag:
        generator_path = "./improvedGAN/improved_celeba_G.tar"

    else:
        generator_path = "./improvedGAN/celeba_G.tar"
            

    G = Generator(100)
    G = torch.nn.DataParallel(G)
    ckp_G = torch.load(generator_path)
    G.load_state_dict(ckp_G['state_dict'])
    G.eval()

    _, trainloader = utils.init_dataloader(config_args, train_file, mode="train")

    fid_list = []

    for i in range(5):
        real_images, fake_images = get_real_batch(trainloader), generate_fake_batch(G)
        FID = calculate_fid(real_images, fake_images)
        fid_list.append(FID.item())
        print(f"Batch {i+1} - FID: {FID.item():.2f}")

    mean_fid = np.mean(fid_list)
    print(f"Mean FID: {mean_fid:.2f}")
