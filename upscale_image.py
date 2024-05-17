import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

# Import the generator model (assuming it's named `generator.py`)
from model.generator import Net as Generator

def load_model(checkpoint_path, device):
    model = Generator(scale_factor=2)  # Ensure the scale_factor matches your training setup
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])  # Load only the generator state
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

def preprocess_image(image_path, device):
    print(f"Preprocessing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Ensure the same normalization as during training
    ])
    image = transform(image).unsqueeze(0).to(device)
    print("Image preprocessed successfully")
    return image

def postprocess_image(tensor):
    print("Postprocessing image")
    # Scale back the values from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    transform = transforms.ToPILImage()
    image = tensor.squeeze(0).cpu()
    image = transform(image)
    print("Image postprocessed successfully")
    return image

def upscale_image(model, image_path, output_path, device):
    lr_image = preprocess_image(image_path, device)
    
    with torch.no_grad():
        print("Generating high-resolution image")
        sr_image = model(lr_image)
    
    print(f"Low-resolution image shape: {lr_image.shape}")
    print(f"High-resolution image shape: {sr_image.shape}")
    print(f"High-resolution image values: {sr_image.min().item()} to {sr_image.max().item()}")

    sr_image = postprocess_image(sr_image)
    sr_image.save(output_path)
    print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    # Define fixed paths
    input_dir = r'C:\Users\Borshon Alfred Gomes\Desktop\University\ESRGCNN_GAN\TestUpscale'
    output_dir = r'C:\Users\Borshon Alfred Gomes\Desktop\University\ESRGCNN_GAN\OutputUpscale'
    checkpoint_path = r'C:\Users\Borshon Alfred Gomes\Desktop\University\ESRGCNN_GAN\checkpoint\checkpoint_epoch_99.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the input directory
    for image_name in os.listdir(input_dir):
        if image_name.endswith('.png'):
            input_path = os.path.join(input_dir, image_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}Upscaled.png")
            upscale_image(model, input_path, output_path, device)
