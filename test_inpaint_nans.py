# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:03:38 2025

@author: etienne
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # or "import imageio" if you have v2
import time
from scipy.interpolate import griddata, interp1d, PchipInterpolator
from inpaint_nans import inpaint_nans  # <-- import your converted function here


############################
# 1) 50% random “artifacts” in an image
############################

# Read and convert image to float
garden = imageio.imread('monet_adresse.jpg')  # Adjust path if needed
G = garden.astype(float)

# Corrupt 50% of the pixels at random
mask = (np.random.rand(*G.shape) < 0.50)
Gnan = G.copy()
Gnan[mask] = np.nan

# Inpaint each color channel with method=2 (or whichever method you want)
G_inpainted = G.copy()  # keep separate copy
for c in range(3):
    G_inpainted[:,:,c] = inpaint_nans(G_inpainted[:,:,c], method=2)

# Display original, corrupted, inpainted
plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.imshow(garden.astype(np.uint8))
plt.title("Garden at Sainte-Adresse (Monet)")

plt.subplot(1,3,2)
plt.imshow(Gnan.astype(np.uint8))
plt.title("Corrupted - 50%")

plt.subplot(1,3,3)
plt.imshow(G_inpainted.astype(np.uint8))
plt.title("Inpainted Garden")
plt.show()

