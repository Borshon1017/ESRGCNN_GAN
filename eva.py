import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchsummary import summary
from thop import profile

# Correct import paths based on the provided directory structure
from esrgcnn.model.generator import Generator as Generator_ESRGCNN_GAN
from esrgcnn.model.esrgcnn_ori import Generator as Generator_ESRGCNN  # Assuming ESRGCNN's generator is defined in esrgcnn_ori.py

# Paths
train_dataset_path = 'dataset/train'
test_dataset_path = 'dataset/test'
esrgcnn_checkpoint = 'checkpoint/esrgcnn.pth'
esrgcnn_gan_checkpoint = 'checkpoint/esrgcnn_gan.pth'

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_dataset = ImageFolder(root=test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
esrgcnn = Generator_ESRGCNN().to(device)
esrgcnn.load_state_dict(torch.load(esrgcnn_checkpoint))
esrgcnn.eval()

esrgcnn_gan = Generator_ESRGCNN_GAN().to(device)
esrgcnn_gan.load_state_dict(torch.load(esrgcnn_gan_checkpoint))
esrgcnn_gan.eval()

# Function to compute PSNR and SSIM
def compute_metrics(output, target):
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    target = target.squeeze().cpu().numpy().transpose(1, 2, 0)
    psnr_value = psnr(target, output)
    ssim_value = ssim(target, output, multichannel=True)
    return psnr_value, ssim_value

# Evaluate models
def evaluate_model(model):
    psnr_list, ssim_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for imgs_lr, imgs_hr in test_loader:
            imgs_lr = imgs_lr.to(device)
            imgs_hr = imgs_hr.to(device)
            sr_imgs = model(imgs_lr)
            psnr_value, ssim_value = compute_metrics(sr_imgs, imgs_hr)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
    end_time = time.time()
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    runtime = end_time - start_time
    return avg_psnr, avg_ssim, runtime

psnr_esrgcnn, ssim_esrgcnn, runtime_esrgcnn = evaluate_model(esrgcnn)
psnr_esrgcnn_gan, ssim_esrgcnn_gan, runtime_esrgcnn_gan = evaluate_model(esrgcnn_gan)

# Count parameters and calculate FLOPs
input_tensor = torch.randn(1, 3, 256, 256).to(device)
params_esrgcnn = sum(p.numel() for p in esrgcnn.parameters() if p.requires_grad)
params_esrgcnn_gan = sum(p.numel() for p in esrgcnn_gan.parameters() if p.requires_grad)
flops_esrgcnn, _ = profile(esrgcnn, inputs=(input_tensor,))
flops_esrgcnn_gan, _ = profile(esrgcnn_gan, inputs=(input_tensor,))

# Print results
print(f'ESRGCNN - PSNR: {psnr_esrgcnn}, SSIM: {ssim_esrgcnn}, Runtime: {runtime_esrgcnn}s, Params: {params_esrgcnn}, FLOPs: {flops_esrgcnn}')
print(f'ESRGCNN-GAN - PSNR: {psnr_esrgcnn_gan}, SSIM: {ssim_esrgcnn_gan}, Runtime: {runtime_esrgcnn_gan}s, Params: {params_esrgcnn_gan}, FLOPs: {flops_esrgcnn_gan}')
