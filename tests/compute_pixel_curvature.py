import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import compute_curvature_pixel
import numpy as np
import imageio.v2 as imageio 
import matplotlib.pyplot as plt


# load video
subject = 'alexandra'
category = 'synthetic'
eccentricity = 'fovea'
movie_id = 6
diameter = 6; # 6, 24, 36
movie_name = 'carnegie-dam'

# read in natural image frames
v_folder = os.path.join('data', 'yoon_stimulus', f'diameter_{diameter:02d}_deg', f'movie{movie_id:02d}-{movie_name}')
im = []
for fname in sorted(os.listdir(v_folder)):
    if category in fname:
        im_path = os.path.join(v_folder, fname)
        im.append(imageio.imread(im_path))

# convert to 3D array and normalize to [0, 1]
I = np.stack(im, axis=-1).astype(np.float64) / 255

# display frames
plt.figure(1)
for iframe in range(I.shape[2]):
    plt.imshow(I[:, :, iframe], cmap='gray')
    plt.axis('off')
    plt.axis('square')
    plt.pause(0.1)

c = compute_curvature_pixel(I)

print(np.rad2deg(np.mean(c)))
