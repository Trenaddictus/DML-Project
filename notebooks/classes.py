from pathlib import Path

from torchvision.models import resnet18, ResNet18_Weights
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from torch.utils.data import Dataset, DataLoader
import timm

class RotationViTGRU(nn.Module):

    def __init__(self, hidden_size=128):
        super().__init__()

        # Pretrained Vision Transformer
        self.encoder = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0
        )

        # Freeze pretrained backbone
        self.encoder.requires_grad_(False)

        # Spatial feature pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 3))

        # 384 channels * 2 * 3 spatial positions
        feature_dim = 384 * 2 * 3

        # Current features + reference difference
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Temporal model
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict cosine and sine
        self.head = nn.Linear(hidden_size, 2)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def extract_features(self, images):

        # Pad to multiples of 16, minimum 128 pixels
        H, W = images.shape[-2:]

        new_h = max(128, ((H + 15) // 16) * 16)
        new_w = max(128, ((W + 15) // 16) * 16)

        images = F.pad(
            images,
            (0, new_w - W, 0, new_h - H),
            mode="replicate"
        )

        # ImageNet normalization
        images = (images - self.mean) / self.std

        # Extract frozen ViT patch embeddings
        with torch.no_grad():
            tokens = self.encoder.forward_features(images)

        # Discard CLS/register tokens, keep spatial patches
        grid_h = new_h // 16
        grid_w = new_w // 16

        patches = tokens[:, -grid_h * grid_w:, :]

        # [T, N, 384] -> [T, 384, grid_h, grid_w]
        patches = patches.transpose(1, 2).reshape(
            images.shape[0], 384, grid_h, grid_w
        )

        features = self.spatial_pool(patches).flatten(1)

        return features

    def forward(self, frames, reference, hidden=None):

        T = frames.shape[0]

        self.encoder.eval()

        # Extract features from frames and reference
        frame_features = self.extract_features(frames)

        reference_features = self.extract_features(
            reference.unsqueeze(0)
        )

        reference_features = reference_features.expand(T, -1)

        # Compare each frame with reference
        difference = frame_features - reference_features

        combined = torch.cat(
            [frame_features, difference],
            dim=1
        )

        # Project to 128 dimensions
        features = self.projection(combined)

        # [T, 128] -> [1, T, 128]
        features = features.unsqueeze(0)

        # Temporal processing
        output, hidden = self.gru(features, hidden)

        # Predict unit vector [cos(theta), sin(theta)]
        predictions = self.head(output.squeeze(0))

        predictions = F.normalize(
            predictions,
            dim=-1,
            eps=1e-6
        )

        return predictions, hidden
    

class RotationCNNGRU(nn.Module):

    def __init__(self, hidden_size=128):
        super().__init__()

        backbone = resnet18(
            weights = ResNet18_Weights.DEFAULT
        )

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2] # remove last 2 layers: average pooling and classification
        )
        for param in self.encoder.parameters():
            param.requires_grad=False # freeze all remaining layers

        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 3)) # changes every feature to [512, 2, 3]

        feature_dim = 512 * 2 * 3

        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # converts large comparison into 128 features per frame. Before entering GRU, it has shape [T, 128]

        # GRU: model rotation over time
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            batch_first=True
        )
        # gru output: [1, T, hidden_size]
        

        # Output: cosine and sine of rotation angle
        self.head = nn.Linear(hidden_size, 2) # angle head

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        ) # image normalization

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        ) # image normalization

    def forward(self, frames, reference, hidden=None):
        # frames: [T, 3, H, W]
        # reference: [3, H, W]

        T = frames.shape[0]

        self.encoder.eval()

        frames = (frames - self.mean) / self.std # normalization

        reference = reference.unsqueeze(0)
        reference = (reference - self.mean) / self.std

        reference_features = self.encoder(reference)
        frame_features = self.encoder(frames)

        reference_features = self.spatial_pool(
            reference_features
        ).flatten(1)

        frame_features = self.spatial_pool(
            frame_features
        ).flatten(1)

        # Compare current frame with reference frame
        reference_features = reference_features.expand(T, -1)

        difference = frame_features - reference_features

        combined = torch.cat(
            [frame_features, difference],
            dim=1
        )

        features = self.projection(combined)

        # Process chronological sequence
        features = features.unsqueeze(0)

        output, hidden = self.gru(
            features,
            hidden
        )

        predictions = self.head(
            output.squeeze(0)
        )

        predictions = F.normalize(
            predictions,
            dim=-1,
            eps=1e-6
        )

        return predictions, hidden

class RotationPositionDataset(Dataset):

    def __init__(
        self,
        split_dir,
        position_file="positions.csv",
        max_side=128,
        transform=None
    ):
        self.csv_files = sorted(
            Path(split_dir).glob("*/rotation_labels/rotation_*.csv")
        )

        self.position_file = position_file
        self.max_side = max_side
        self.transform = transform

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):

        csv_path = self.csv_files[idx]
        video_dir = csv_path.parent.parent

        rotations = pd.read_csv(csv_path)

        # Match object positions to rotation labels by filename
        positions = (
            pd.read_csv(video_dir / self.position_file)
            .set_index("filename")
            .reindex(rotations["filename"])
        )

        xy = positions[["cx", "cy"]].to_numpy(dtype=np.float32)

        images = []
        original_xy = []
        rotated_xy = []
        valid = []

        for i, row in enumerate(rotations.itertuples(index=False)):

            image = cv2.imread(str(video_dir / "input" / row.filename))
            h, w = image.shape[:2]

            M = cv2.getRotationMatrix2D(
                (w / 2, h / 2),
                row.angle_deg,
                row.zoom
            )

            # Rotate image
            image = cv2.warpAffine(image, M, (w, h))

            # Rotate object position using the same matrix
            point = xy[i]
            point_rot = M @ np.array([point[0], point[1], 1.0])

            # Normalize positions to image coordinates [0,1]
            original_xy.append(point / [w, h])
            rotated_xy.append(point_rot / [w, h])

            valid.append(
                np.isfinite(point_rot).all()
                and 0 <= point_rot[0] < w
                and 0 <= point_rot[1] < h
            )

            # Resize image
            scale = self.max_side / max(h, w)
            new_size = (
                max(8, round(w * scale / 8) * 8),
                max(8, round(h * scale / 8) * 8)
            )

            image = cv2.resize(image, new_size)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        frames = torch.from_numpy(
            np.stack(images)
        ).permute(0, 3, 1, 2).float() / 255.0

        return {
            "frames": frames,
            "angles": torch.tensor(
                rotations["angle_deg"].to_numpy(), dtype=torch.float32
            ),
            "pos_original": torch.tensor(original_xy, dtype=torch.float32),
            "pos_rotated": torch.tensor(rotated_xy, dtype=torch.float32),
            "pos_valid": torch.tensor(valid, dtype=torch.bool),
            "zoom": torch.tensor(
                rotations["zoom"].to_numpy(), dtype=torch.float32
            )
        }
