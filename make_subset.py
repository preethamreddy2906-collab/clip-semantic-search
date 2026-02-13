import os
import random
import shutil
import torch

# paths
img_dir = "data/images"
subset_dir = "data/images_subset"

os.makedirs(subset_dir, exist_ok=True)

# load original img_names
img_names = torch.load("embeddings/img_names_new.pt")

# choose subset
subset_size = 200
subset_names = random.sample(img_names, subset_size)

# copy files
for name in subset_names:
    src = os.path.join(img_dir, name)
    dst = os.path.join(subset_dir, name)
    if os.path.exists(src):
        shutil.copy(src, dst)

# save subset list
torch.save(subset_names, "embeddings/img_names_subset.pt")

print("Subset created:", len(subset_names))
