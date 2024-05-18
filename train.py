import os
import importlib
import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import TrainDataset 

class Config:
    def __init__(self):
        self.model = 'generator'
        self.hr_data_dir = 'dataset/DIV2K/DIV2K_train_HR'  
        self.lr_data_dir = 'dataset/DIV2K/DIV2K_train_LR_bicubic' 
        self.scale = 2
        self.ckpt_path = 'checkpoint'
        self.batch_size = 16
        self.epoch = 100
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.patch_size = 64

cfg = Config()

def main(cfg):
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
  
    Generator = importlib.import_module("model.generator").Net
    Discriminator = importlib.import_module("model.discriminator").Net
    

    generator = Generator(scale_factor=cfg.scale)
    discriminator = Discriminator()

    # Print model architectures
    print(generator)
    print(discriminator)

    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_content = nn.MSELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    # Prepare data
    print("Loading datasets...")
    train_dataset = TrainDataset(cfg.hr_data_dir, os.path.join(cfg.lr_data_dir, f'X{cfg.scale}'), scale=cfg.scale, size=cfg.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"Loaded {len(train_dataset)} samples from dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    print("Models moved to device:", device)

    # Training loop
    for epoch in range(cfg.epoch):
        print(f"Starting epoch {epoch+1}/{cfg.epoch}...")
        for i, batch in enumerate(train_loader):
            print(f"Processing batch {i+1}/{len(train_loader)}...")
            imgs_lr, imgs_hr = batch
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)
            
            valid = torch.ones((imgs_lr.size(0), 1)).to(device)
            fake = torch.zeros((imgs_lr.size(0), 1)).to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            gen_hr = generator(imgs_lr)
            loss_content = criterion_content(gen_hr, imgs_hr)
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            loss_G = loss_content + 0.001 * loss_GAN
            loss_G.backward()
            optimizer_G.step()
            print(f"Generator loss: {loss_G.item()}")

            # Train Discriminator
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            print(f"Discriminator loss: {loss_D.item()}")

        # Save checkpoint
        print("Saving checkpoint...")
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, os.path.join(cfg.ckpt_path, f"checkpoint_epoch_{epoch}.pth"))
        print(f"Checkpoint saved for epoch {epoch+1}.")

if __name__ == "__main__":
    main(cfg)
