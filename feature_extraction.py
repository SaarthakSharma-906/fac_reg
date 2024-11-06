
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
import torch.nn.functional as F
from preprocessing import Whitening

class FaceFeaturesExtractor:
    def __init__(self):
        # Tune MTCNN thresholds
        self.aligner = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.8])
        self.facenet_preprocess = transforms.Compose([Whitening()])
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_features(self, img):
        bbs, _ = self.aligner.detect(img)
        if bbs is None or len(bbs) == 0:
            print("No faces detected.")
            return None, None

        faces = torch.stack([extract_face(img, bb) for bb in bbs])
        if faces.size(0) == 0:
            print("Empty tensor of faces.")
            return None, None

        prewhitened_faces = self.facenet_preprocess(faces)
        embeddings = self.facenet(prewhitened_faces).detach()

        # Apply L2 normalization to embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).numpy()

        return bbs, embeddings

    def __call__(self, img):
        return self.extract_features(img)
