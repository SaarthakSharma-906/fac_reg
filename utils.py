import numpy as np
from torchvision import transforms
from PIL import Image

def dataset_to_embeddings(dataset, features_extractor, preprocessor):
    transform = transforms.Compose([
        preprocessor,  # Add any additional preprocessing like resizing, etc.
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f"Processing {img_path}")
        img = transform(Image.open(img_path).convert('RGB'))
        _, embedding = features_extractor(img)
        if embedding is None:
            print(f"Could not find face on {img_path}")
            continue
        if embedding.shape[0] > 1:
            print(f"Multiple faces detected for {img_path}, taking one with highest probability")
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


