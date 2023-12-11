import os
import clip
import torch
#from torchvision.datasets import CIFAR100
import pandas as pd
import PIL
import sys


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)
print('Model input resulution: ', model.visual.input_resolution)
print('Context length resulution: ', model.context_length)
# Download the dataset
#cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
data = pd.read_csv('VisA/split_csv/1cls.csv')
img_path = os.path.join('VisA', data.image[8500])
img = PIL.Image.open(img_path, mode='r')
if img is None:
    sys.exit("Could not read the image.")
img.show()

labels = data.label.unique()
print(labels)

# Prepare the inputs
#image, class_id = cifar100[3637]
image_input = preprocess(img).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)


# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(2)


# Print the result
print("\nTop predictions:\n")
print('True label = ', data.label[8500])
for value, index in zip(values, indices):
    print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")
