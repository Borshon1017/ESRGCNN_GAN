import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale, size):
        print(f"Initializing TrainDataset with HR dir: {hr_dir}, LR dir: {lr_dir}, scale: {scale}, size: {size}")
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.size = size
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        print(f"Found {len(self.hr_images)} HR images and {len(self.lr_images)} LR images")
        self.transform = transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print("Transformations initialized")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        print(f"Loading image pair {idx}")
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        print(f"HR image path: {hr_image_path}")
        print(f"LR image path: {lr_image_path}")
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        print(f"Images loaded, applying transformations")
        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)
        print(f"Transformations applied")
        return lr_image, hr_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        return self.main(x)

# Load the pre-trained generator model
generator = Generator()
checkpoint_path = os.path.join('checkpoint', 'checkpoint_epoch_99.pth')
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path)
generator.load_state_dict(checkpoint['generator'])
generator.eval()
print("Generator model loaded and set to evaluation mode")

# Define dataset paths relative to the current working directory
hr_dir = os.path.join('dataset', 'DIV2K', 'DIV2K_train_HR')
lr_dir_x2 = os.path.join('dataset', 'DIV2K', 'DIV2K_train_LR_bicubic', 'X2')
lr_dir_x3 = os.path.join('dataset', 'DIV2K', 'DIV2K_train_LR_bicubic', 'X3')
lr_dir_x4 = os.path.join('dataset', 'DIV2K', 'DIV2K_train_LR_bicubic', 'X4')

# Create datasets and data loaders for each scale factor
print("Creating datasets and data loaders")
test_dataset_x2 = TrainDataset(hr_dir=hr_dir, lr_dir=lr_dir_x2, scale=2, size=256)
test_dataset_x3 = TrainDataset(hr_dir=hr_dir, lr_dir=lr_dir_x3, scale=3, size=256)
test_dataset_x4 = TrainDataset(hr_dir=hr_dir, lr_dir=lr_dir_x4, scale=4, size=256)

test_loader_x2 = DataLoader(test_dataset_x2, batch_size=1, shuffle=False)
test_loader_x3 = DataLoader(test_dataset_x3, batch_size=1, shuffle=False)
test_loader_x4 = DataLoader(test_dataset_x4, batch_size=1, shuffle=False)
print("Datasets and data loaders created")

def evaluate_model(generator, dataloader):
    psnr_values = []
    ssim_values = []

    for lr, hr in dataloader:
        lr = lr.cuda() if torch.cuda.is_available() else lr
        hr = hr.cuda() if torch.cuda.is_available() else hr
        sr = generator(lr)
        sr = sr.detach().cpu().numpy().squeeze()
        hr = hr.detach().cpu().numpy().squeeze()

        psnr_value = psnr(hr, sr, data_range=hr.max() - hr.min())
        
        # Adjust win_size based on image size
        min_dim = min(hr.shape[0], hr.shape[1])
        win_size = min(7, min_dim)
        ssim_value = ssim(hr, sr, win_size=win_size, data_range=hr.max() - hr.min(), channel_axis=-1)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

# Evaluate the model on the test datasets
print("Evaluating model on test datasets")
psnr_x2, ssim_x2 = evaluate_model(generator, test_loader_x2)
psnr_x3, ssim_x3 = evaluate_model(generator, test_loader_x3)
psnr_x4, ssim_x4 = evaluate_model(generator, test_loader_x4)

print(f"X2: PSNR = {psnr_x2}, SSIM = {ssim_x2}")
print(f"X3: PSNR = {psnr_x3}, SSIM = {ssim_x3}")
print(f"X4: PSNR = {psnr_x4}, SSIM = {ssim_x4}")

def visualize_results(generator, dataloader, save_path):
    for i, (lr, hr) in enumerate(dataloader):
        lr = lr.cuda() if torch.cuda.is_available() else lr
        sr = generator(lr)
        sr = sr.detach().cpu().numpy().squeeze()
        hr = hr.detach().cpu().numpy().squeeze()
        lr = lr.detach().cpu().numpy().squeeze()

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(lr.transpose(1, 2, 0))
        plt.title('Low Resolution')

        plt.subplot(1, 3, 2)
        plt.imshow(hr.transpose(1, 2, 0))
        plt.title('High Resolution')

        plt.subplot(1, 3, 3)
        plt.imshow(sr.transpose(1, 2, 0))
        plt.title('Super Resolved')

        plt.savefig(f"{save_path}/comparison_{i}.png")
        plt.close()

# Ensure the results directory exists
save_path = 'results'
os.makedirs(save_path, exist_ok=True)

# Visualize results for each scale factor
print("Visualizing results")
visualize_results(generator, test_loader_x2, save_path)
visualize_results(generator, test_loader_x3, save_path)
visualize_results(generator, test_loader_x4, save_path)
print("Done visualizing results")
