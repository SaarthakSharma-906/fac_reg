import joblib
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from train import ExifOrientationNormalize, FaceFeaturesExtractor

def cosine_similarity_match(embedding, reference_embeddings, threshold=0.4):
    similarities = cosine_similarity(embedding.reshape(1, -1), reference_embeddings)
    max_similarity = similarities.max()
    # Return both the class index and the similarity score
    if max_similarity >= threshold:
        return np.argmax(similarities), max_similarity
    else:
        return "Unknown", max_similarity

def main(image_path, model_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')

    # Normalize image orientation
    normalizer = ExifOrientationNormalize()
    img = normalizer(img)

    # Load the model components
    classifier, idx_to_class = joblib.load(model_path)

    # Instantiate the feature extractor
    feature_extractor = FaceFeaturesExtractor()

    # Extract features for multiple faces
    bbs, embeddings = feature_extractor.extract_features(img)

    if embeddings is None or len(embeddings) == 0:
        print("No faces detected.")
        return

    # Get the reference embeddings from the classifier
    reference_embeddings = classifier.coef_

    # Set a higher threshold for cosine similarity
    threshold = 0.4

    # Make predictions for each face using cosine similarity
    for i, bb in enumerate(bbs):
        idx, similarity = cosine_similarity_match(embeddings[i], reference_embeddings, threshold)
        print(f"Detected face with bounding box: {bb}")
        print(f"Cosine similarity: {similarity}")
        
        if idx != "Unknown":
            top_prediction = idx_to_class[idx]
            print(f"Top prediction: {top_prediction} with similarity {similarity}")
        else:
            print(f"Low confidence for face at {bb}, marked as 'Unknown'. The similarity is {similarity}")

if __name__ == '__main__':
    image_path = r"C:\Users\Saarthak\Desktop\face_recognition\test_images\WIN_20240924_19_10_39_Pro.jpg"  # Replace with your image path
    model_path = r'C:\Users\Saarthak\Desktop\face_recognition\face_recogniser_on_aug_data.pkl'  # Replace with your model path
    main(image_path, model_path)



