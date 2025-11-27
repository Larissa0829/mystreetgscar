import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import torchvision.transforms as transforms

# Helper function to calculate FID
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Calculates the Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2

    # Ensure sigma1 and sigma2 are at least 2D matrices
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Product of covariance matrices
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Main FID function
def fid(image1, image2, device="cuda"):
    """
    Compute the FID between two images.
    
    Parameters:
        image1 (torch.Tensor): Image tensor with shape (C, H, W), values in range [0, 1].
        image2 (torch.Tensor): Image tensor with shape (C, H, W), values in range [0, 1].
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        float: FID score.
    """
    # Load pre-trained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    
    # Transform images to match InceptionV3 input requirements
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image1 = transform(image1)
    image2 = transform(image2)
    
    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract features from both images
        features1 = model(image1).cpu().numpy().reshape(-1, 1)
        features2 = model(image2).cpu().numpy().reshape(-1, 1)
    
    # Compute mean and covariance for both feature sets
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
    
    # Calculate Frechet Distance
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score
