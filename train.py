import os
import sys
import importlib
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import TrainDataset  # Ensure you import your dataset

class Config:
    def __init__(self):
        self.model = 'generator'
        self.hr_data_dir = 'dataset/DIV2K/DIV2K_train_HR'  # Adjust the path to your HR dataset folder
        self.lr_data_dir = 'dataset/DIV2K/DIV2K_train_LR_bicubic'  # Adjust the path to your LR dataset folder
        self.scale = 2
        self.ckpt_path = 'checkpoint'
        self.sample_dir = 'samples'
        self.batch_size = 16
        self.epoch = 100
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.patch_size = 64

cfg = Config()

def main(cfg):
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    # Import
    Generator = importlib.import_module("model.generator").Net
    Discriminator = importlib.import_module("model.discriminator").Net
    
   
    generator = Generator(scale_factor=cfg.scale)
    discriminator = Discriminator()

    # loss functions
    criterion_GAN = nn.BCELoss()
    criterion_content = nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    # Prepare data
    train_dataset = TrainDataset(cfg.hr_data_dir, cfg.lr_data_dir, scale=cfg.scale, size=cfg.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Training loop
    for epoch in range(cfg.epoch):
        for i, batch in enumerate(train_loader):
            imgs_lr, imgs_hr = batch
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)
            
            valid = torch.ones((imgs_lr.size(0), 1)).to(device)
            fake = torch.zeros((imgs_lr.size(0), 1)).to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            gen_hr = generator(imgs_lr)
            loss_content = criterion_content(gen_hr, imgs_hr)
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{cfg.epoch}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

        # Save checkpoint
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, os.path.join(cfg.ckpt_path, f"checkpoint_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main(cfg)
