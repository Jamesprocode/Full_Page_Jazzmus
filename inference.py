"""
Inference pipeline for full-page jazz leadsheet recognition.

Steps:
1. Load trained model checkpoint
2. Process image (resize, normalize)
3. Generate predictions token-by-token
4. Decode to kern format
5. Display/save results
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.dataset.tokenizer import untokenize


class FullPageInference:
    """Inference pipeline for full-page jazz leadsheet recognition."""

    def __init__(self, checkpoint_path, device="cuda"):
        """
        Initialize inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on (cuda or cpu)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model with same config as checkpoint
        # Note: You'll need to load the config from somewhere
        # For now, we'll load it manually
        self.model = SMT_Trainer.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)

        print("✓ Model loaded successfully")

    def preprocess_image(self, image_path, max_height=1024, max_width=2048):
        """
        Preprocess image for inference.

        Args:
            image_path: Path to input image
            max_height: Maximum image height
            max_width: Maximum image width

        Returns:
            torch.Tensor: Preprocessed image
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.array(image_path)

        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize with aspect ratio preservation
        height, width = img.shape
        aspect_ratio = width / height

        # Fit to max dimensions
        if height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = height
            new_width = width

        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)

        # Resize
        img = cv2.resize(img, (new_width, new_height))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Pad to max size
        padded = torch.zeros(1, 1, max_height, max_width)
        padded[:, :, :new_height, :new_width] = img_tensor[:, :, :new_height, :new_width]

        print(f"✓ Image preprocessed: {(new_height, new_width)} -> {padded.shape}")

        return padded.to(self.device)

    def predict(self, image_path, return_probs=False):
        """
        Predict on full-page image.

        Args:
            image_path: Path to input image
            return_probs: Return token probabilities

        Returns:
            dict: Prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        # Generate predictions
        print("Generating predictions...")
        with torch.no_grad():
            predicted_tokens, logits = self.model.model.predict(input=image_tensor[0])

        # Decode tokens to string
        token_strs = [self.model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
        prediction_str = untokenize(token_strs)

        results = {
            "tokens": token_strs,
            "prediction": prediction_str,
            "num_tokens": len(predicted_tokens),
        }

        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
            results["logits"] = logits.cpu().numpy()
            results["top_probs"] = top_probs.cpu().numpy()
            results["top_indices"] = top_indices.cpu().numpy()

        return results

    def save_result(self, result, output_path):
        """
        Save prediction result to file.

        Args:
            result: Prediction result dictionary
            output_path: Path to save result
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['prediction'])

        print(f"✓ Result saved to: {output_path}")

    def display_result(self, result):
        """
        Display prediction result.

        Args:
            result: Prediction result dictionary
        """
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Number of tokens: {result['num_tokens']}")
        print(f"\nPredicted kern (first 500 chars):")
        print(result['prediction'][:500])
        if len(result['prediction']) > 500:
            print("...")
        print("=" * 60)


def run_inference(
    checkpoint_path,
    image_path,
    output_path=None,
    device="cuda",
):
    """
    Run inference on a single image.

    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to input image
        output_path: Optional path to save prediction
        device: Device to run on
    """
    # Initialize inference
    inference = FullPageInference(checkpoint_path, device=device)

    # Predict
    result = inference.predict(image_path)

    # Display result
    inference.display_result(result)

    # Save if output path provided
    if output_path:
        inference.save_result(result, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(run_inference)
