class RotationDataset(torch.utils.data.Dataset):

    def __getitem__(self, idx):

        # Load original video frames
        frames = self.load_sequence(idx)

        # Generate new synthetic rotations
        result = rotate_sequence(
            frames,
            max_angle_deg=15,
            max_step_deg=2,
            seed=None
        )

        # Convert images to PyTorch format
        images = torch.from_numpy(
            result["frames"]
        ).permute(0, 3, 1, 2).float() / 255.0

        angles = torch.tensor(
            result["angles_deg"],
            dtype=torch.float32
        )

        return images, angles

