# Facial Reconstruction with Super-Resolution using Latent Diffusion model

This project is a proof-of-concept (POC) for enhancing low-resolution CCTV images using a deep learning-based super-resolution model. The goal is to reconstruct human faces from low-quality footage for better identification.

## Key Features
- Uses a diffusion-based super-resolution model to upscale low-resolution images. LDM
- Paired low- and high-resolution image dataset for training and testing.
- Visualization of the output compared to the input.

## Dataset
The dataset consists of low-resolution and high-resolution image pairs stored in the directories:
- `high_res/`: Contains high-resolution images.
- `low_res/`: Contains low-resolution images.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
