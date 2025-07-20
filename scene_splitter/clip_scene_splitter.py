import numpy as np
import torch
import cv2
from PIL import Image
import clip
from tqdm import tqdm
from sklearn.cluster import KMeans

class CLIPSceneSplitter:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def extract_frames(self, video_path, every_n_frames=15):
        cap = cv2.VideoCapture(video_path)
        frames = []
        timestamps = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                timestamps.append(frame_idx / fps)
            frame_idx += 1

        cap.release()
        return frames, timestamps

    def compute_embeddings(self, frames):
        embeddings = []
        with torch.no_grad():
            for frame in tqdm(frames, desc="Computing embeddings"):
                image = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image)
                embeddings.append(embedding.cpu().numpy()[0])
        return np.stack(embeddings)

    def cluster_embeddings(self, embeddings, n_clusters=8):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels

    def get_scenes(self, cluster_labels, timestamps):
        scenes = []
        start = 0
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i - 1]:
                scenes.append((timestamps[start], timestamps[i]))
                start = i
        scenes.append((timestamps[start], timestamps[-1]))
        return scenes

    def detect_scenes(self, video_path, every_n_frames=15, n_clusters=8):
        frames, timestamps = self.extract_frames(video_path, every_n_frames)
        embeddings = self.compute_embeddings(frames)
        labels = self.cluster_embeddings(embeddings, n_clusters)
        scenes = self.get_scenes(labels, timestamps)
        return scenes

