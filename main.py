import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import LDMSuperResolutionPipeline
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

high_res_dir = "/kaggle/working/high_res/high_res"
low_res_dir = "/kaggle/working/low_res/final_low"

class ImagePairDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None):
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.transform = transform
        self.image_pairs = [(os.path.join(low_res_dir, f), os.path.join(high_res_dir, f))
                            for f in os.listdir(low_res_dir) if f in os.listdir(high_res_dir)]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_res_path, high_res_path = self.image_pairs[idx]
        low_res_image = Image.open(low_res_path).convert("RGB")
        
        if self.transform:
            low_res_image = self.transform(low_res_image)
        
        return low_res_image, low_res_path


transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
])


dataset = ImagePairDataset(high_res_dir, low_res_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model_id = "CompVis/ldm-super-resolution-4x-openimages"
ldm_pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
ldm_pipeline = ldm_pipeline.to(device)

# Fine-tuning 
@torch.no_grad()
def fine_tune_step(image):
    return ldm_pipeline(image, num_inference_steps=50, eta=1).images[0]


def process_batch(batch_size=4):
    res_originals = []
    res_outputs = []
    
    for i, (low_res, low_res_path) in enumerate(dataloader):
        low_res = low_res.to(device)
        res_image = fine_tune_step(low_res)
        res_originals.append(low_res.cpu().squeeze(0).permute(1, 2, 0))
        res_outputs.append(res_image)
        torch.cuda.empty_cache()
        
        if (i + 1) % batch_size == 0:
            yield res_originals, res_outputs
            res_originals = []
            res_outputs = []
            gc.collect()  

def plot_results(originals, outputs, num_samples=3):
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(originals))):
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(originals[i])
        plt.title("Low-Resolution")
        plt.axis("off")
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(outputs[i])
        plt.title("Super-Resolved Output")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


for batch_originals, batch_outputs in process_batch(batch_size=4):
    plot_results(batch_originals, batch_outputs, num_samples=3)
    break  