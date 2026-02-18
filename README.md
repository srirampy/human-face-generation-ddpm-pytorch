# Human Face Generation using Denoising Diffusion Probabilistic Models (DDPM)

## Overview
This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in **PyTorch** to generate realistic human faces using the **CelebA dataset**.

The model learns to gradually convert pure Gaussian noise into a realistic human face by iteratively predicting and removing noise.  
Unlike GANs, diffusion models are stable to train and produce high-quality images.

This repository demonstrates the complete generative pipeline:

- Forward diffusion (noise corruption)
- Reverse diffusion (image generation)
- UNet architecture with time embeddings
- Training progression visualization

---

## Training Progress
Below is the evolution of the same noise sample across epochs:

![Training Progress](training_progress.gif)

The GIF shows:
**noise → blobs → face structure → realistic human**

---

## Dataset
**CelebA (Large-scale CelebFaces Attributes Dataset)**  
~200,000 aligned celebrity face images.

### Preprocessing
- Resized to **64×64**
- Normalized to **[-1, 1]**
- Random horizontal flip augmentation

> Dataset is not uploaded due to size (~1.3GB). It is publicly available online.

For convenience, I have already prepared and uploaded the dataset on Kaggle.  
You can directly attach it to your Kaggle notebook and start training without manually downloading or preprocessing.

**Kaggle Dataset Link:**  
https://www.kaggle.com/datasets/ram3288/celeba-faces-dataset
---

## Model Architecture

The model uses a **UNet backbone** with sinusoidal timestep embeddings.

The network is trained to predict the noise `ε` added to an image at timestep `t`.

Mathematically, the model learns:

εθ(xt, t) → predicted noise

Where:
- `xt` = noisy image at timestep `t`
- `t` = diffusion step (noise level)
- `ε` = Gaussian noise added to the image
- `εθ` = neural network prediction


### Training Objective
Mean Squared Error (MSE) between:
- true noise
- predicted noise

### Key Components
- Diffusion scheduler (1000 timesteps)
- Sinusoidal positional embeddings
- Encoder–decoder UNet
- Reverse diffusion sampling

---

## How Diffusion Works (Intuition)

1. Start with a real image
2. Gradually add Gaussian noise until it becomes pure noise
3. Train the network to predict the noise at each step
4. During inference, start from random noise and repeatedly remove noise

Result:
> Random noise transforms into a realistic human face.

---

## Training Details

| Parameter | Value |
|--------|------|
| Image size | 64×64 |
| Diffusion steps | 1000 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Loss | Mean Squared Error |
| GPU | NVIDIA Tesla T4 (Kaggle) |

Training time: **~15–20 minutes per epoch**

---

## Results
- After ~8 epochs → visible face structure
- After ~10–12 epochs → recognizable faces
- After ~15 epochs → realistic faces

The model learns the **probability distribution of human faces** rather than memorizing images.

---

## Running the Project

1. Download the CelebA dataset
2. Place images inside:

```
img_align_celeba/
```

3. Open the notebook:

```
celeba_diffusion.ipynb
```

4. Train the model
5. Run the sampling function to generate faces

---

## Output
The model generates **completely new humans** that do not exist in the dataset.

This demonstrates a generative probabilistic model learning the underlying distribution of real-world images.

---

## Future Improvements
- Conditional generation (smile, gender, glasses)
- Higher resolution (128×128)
- DDIM fast sampling
- Classifier-free guidance
- Text-to-image generation

---

## Reference
Ho et al., 2020 — *Denoising Diffusion Probabilistic Models*

---

## Author
**Marpina Sai Sri Ram**

This project is part of my research exploration in **Generative AI and Deep Learning**.
