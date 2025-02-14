
"""Feature map extraction with the specifc layers we want"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os 
from PIL import Image 
import requests

 # Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)


class AlexNet(nn.Module):
	def __init__(self):
		"""Select the desired layers and create the model."""
		super(AlexNet, self).__init__()
		self.alex_feats = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features

	def forward(self, x):
		"""Extract the feature maps."""
		features = []
		for name, layer in self.alex_feats._modules.items():
			x = layer(x)
			if name == '0': # refers to first conv layer
				return x

# Load AlexNet pre-trained on ImageNet
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
alexnet.eval()


# Use the appropriate transformations to preprocess your dataset to match AlexNet's input size (224x224 RGB images):
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# load and extract feature maps
# image directories
img_set_dir = "/Users/elvissmith/Desktop/training_images"
img_partitions = [p for p in os.listdir(img_set_dir) if not p.startswith(".")]  # Exclude hidden files

#iterate through each subfolder (class labels) in training_images
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
    #iterates through each jpeg in each subfolder
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.lower().endswith((".jpg", ".jpeg")):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	
 	# Create the saving directory if not existing
	save_dir = os.path.join('dnn_feature_maps', 'alexnet', 'pretrained-True', p)
	os.makedirs(save_dir, exist_ok=True)
	

	# Extract and save the feature maps
	for i, image in enumerate(image_list):
		# Generate the save path
		file_name = p + '_' + format(i + 1, '07') + '.npy'
		save_path = os.path.join(save_dir, file_name)

		# Skip if the feature map already exists
		if os.path.exists(save_path):
			print(f"Feature map already exists for {image}, skipping.")
			continue

		try:
			img = Image.open(image).convert('RGB')  # Attempt to open and process the image
			input_img = transform(img).unsqueeze(0)

			if torch.cuda.is_available():
				input_img = input_img.cuda()

			# Extract feature map from the first layer
			feature_map = alexnet.features[0](input_img)

			# Save the feature map
			np.save(save_path, feature_map.data.cpu().numpy())

		except (PIL.UnidentifiedImageError, IOError) as e:
			print(f"Skipping unreadable image: {image}. Error: {e}")



