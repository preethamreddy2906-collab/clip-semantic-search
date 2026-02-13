import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
import faiss

#device="cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
clip_model,clip_preprocess=clip.load("ViT-B/32",device=device)


img_features=torch.load("embeddings/img_features_new.pt",map_location=torch.device("cpu")).float()
captions_features=torch.load("embeddings/captions_features_new.pt",map_location=torch.device("cpu")).float()
processed_captions=torch.load("embeddings/processed_captions_new.pt",map_location=torch.device("cpu"))
faiss_index = faiss.read_index("embeddings/image_index.faiss")

def similarity(text_features,image_features):
  text_norm=text_features/text_features.norm(dim=-1,keepdim=True)
  img_norm=image_features/image_features.norm(dim=-1,keepdim=True)
  return text_norm @ img_norm.T


def zero_shot_classifier(clip_model,clip_preprocess,image,captions_features,device=device):
  image=clip_preprocess(image).unsqueeze(0).to(device)
  with torch.no_grad():
    image_features=clip_model.encode_image(image).float()
  similarity_cap=similarity(captions_features,image_features)
  similarity_cap=similarity_cap.cpu().detach().numpy()
  idx=np.argmax(similarity_cap)
  return idx

def generate_caption(img,clip_model,clip_preprocess):
  idx=zero_shot_classifier(clip_model,clip_preprocess,img,captions_features)
  return processed_captions[idx]




class ImageRetrieval:
    def __init__(self, clip_model, faiss_index, device):
        self.clip_model = clip_model
        self.device = device
        self.index=faiss_index

    @torch.no_grad()
    def retrieve(self, query, k: int = 2):
        query = clip.tokenize([query]).to(self.device)
        query_features = self.clip_model.encode_text(query)
        query_features /= query_features.norm(dim=1, keepdim=True)
        query_features=query_features.float()
        query_np=query_features.cpu().numpy().astype("float32")
        _, indices = self.index.search(query_np, k)
        return indices[0].tolist()


