#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python matplotlib numpy torch torchvision kaggle')


# In[3]:


get_ipython().system('pip install opencv-python matplotlib numpy')


# In[1]:


get_ipython().system('pip install numpy pandas matplotlib opencv-python pillow torch torchvision pycocotools')


# In[1]:


get_ipython().system('pip install numpy==1.26.4 matplotlib pillow pycocotools opencv-python')


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO


# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


dataset_path = r"C:\Users\sreel\Desktop\VisionExtract\supervisely_person_clean\person_segmentation"

image_folder = os.path.join(dataset_path, "images")

mask_folder = os.path.join(dataset_path, "masks")

print("Images:", len(os.listdir(image_folder)))
print("Masks:", len(os.listdir(mask_folder)))


# In[13]:


image_files = sorted(os.listdir(image_folder))

image_files = image_files[55:60]

print(image_files)


# In[14]:


for i, file_name in enumerate(image_files):

    image_path = os.path.join(image_folder, file_name)

    mask_path = os.path.join(mask_folder, file_name)

    # Load image

    image = Image.open(image_path).convert("RGB")

    image_np = np.array(image)

    # Load mask

    mask = Image.open(mask_path).convert("L")

    mask_np = np.array(mask)

    # Convert mask to binary

    binary_mask = (mask_np > 0).astype(np.uint8)

    # Isolate subject

    isolated = image_np * binary_mask[:, :, None]

    # Show results

    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title(f"Original Image {i+1}")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title(f"Mask {i+1}")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(isolated)
    plt.title(f"Isolated Subject {i+1}")
    plt.axis("off")

    plt.show()


# In[ ]:




