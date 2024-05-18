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
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.size = size
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.transform = transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)
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


generator = Generator()
generator.load_state_dict(torch.load('checkpoint/checkpoint_epoch_99.pth'))
generator.eval()


test_dataset_set5 = TrainDataset(hr_dir='dataset/Set5/x4', lr_dir='dataset/Set5/x4', scale=4, size=256)
test_dataset_set14 = TrainDataset(hr_dir='dataset/Set14/x4', lr_dir='dataset/Set14/x4', scale=4, size=256)
test_dataset_bsd100 = TrainDataset(hr_dir='dataset/BSD100/x4', lr_dir='dataset/BSD100/x4', scale=4, size=256)
test_dataset_urban100 = TrainDataset(hr_dir='dataset/Urban100/x4', lr_dir='dataset/Urban100/x4', scale=4, size=256)

test_loader_set5 = DataLoader(test_dataset_set5, batch_size=1, shuffle=False)
test_loader_set14 = DataLoader(test_dataset_set14, batch_size=1, shuffle=False)
test_loader_bsd100 = DataLoader(test_dataset_bsd100, batch_size=1, shuffle=False)
test_loader_urban100 = DataLoader(test_dataset_urban100, batch_size=1, shuffle=False)

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
        ssim_value = ssim(hr, sr, multichannel=True)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

psnr_set5, ssim_set5 = evaluate_model(generator, test_loader_set5)
psnr_set14, ssim_set14 = evaluate_model(generator, test_loader_set14)
psnr_bsd100, ssim_bsd100 = evaluate_model(generator, test_loader_bsd100)
psnr_urban100, ssim_urban100 = evaluate_model(generator, test_loader_urban100)

print(f"Set5: PSNR = {psnr_set5}, SSIM = {ssim_set5}")
print(f"Set14: PSNR = {psnr_set14}, SSIM = {ssim_set14}")
print(f"BSD100: PSNR = {psnr_bsd100}, SSIM = {ssim_bsd100}")
print(f"Urban100: PSNR = {psnr_urban100}, SSIM = {ssim_urban100}")

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


save_path = 'results'
os.makedirs(save_path, exist_ok=True)

visualize_results(generator, test_loader_set5, save_path)
visualize_results(generator, test_loader_set14, save_path)
visualize_results(generator, test_loader_bsd100, save_path)
visualize_results(generator, test_loader_urban100, save_path)
