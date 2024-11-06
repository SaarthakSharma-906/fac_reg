
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torchvision import datasets
from collections import defaultdict
from preprocessing import ExifOrientationNormalize  # Import preprocessors
from feature_extraction import FaceFeaturesExtractor  # Import feature extractor
from utils import dataset_to_embeddings  # Import utilities

def filter_dataset(dataset, class_limit=10):
    """Limits each class to a maximum number of images (class_limit)."""
    class_sample_counts = defaultdict(int)
    filtered_samples = []

    for img_path, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_sample_counts[class_name] < class_limit:
            filtered_samples.append((img_path, label))
            class_sample_counts[class_name] += 1

    # Update the dataset with filtered samples
    dataset.samples = filtered_samples
    dataset.targets = [label for _, label in filtered_samples]
    return dataset

def train_model():
    # Define the dataset path
    dataset_path = r"C:\Users\Saarthak\Desktop\face_recognition\augmented_data"

    # Load the dataset
    dataset = datasets.ImageFolder(dataset_path)

    # Limit each class to 5 images
    dataset = filter_dataset(dataset, class_limit=5)

    # Initialize the FaceFeaturesExtractor and preprocessors
    features_extractor = FaceFeaturesExtractor()
    preprocessor = ExifOrientationNormalize()

    # Extract embeddings and labels from the filtered dataset
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor, preprocessor)

    # Train a logistic regression classifier on the embeddings
    clf = LogisticRegression(multi_class='multinomial', C=10, max_iter=10000, class_weight='balanced')
    clf.fit(embeddings, labels)

    # Evaluate the classifier
    target_names = dataset.classes  # Correct way to get target names
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=target_names))

    # Save the trained model to a file
    model_data = (clf, dataset.classes)
    model_path = os.path.join(os.getcwd(), 'face_recogniser_on_aug_data_1.pkl')
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()

