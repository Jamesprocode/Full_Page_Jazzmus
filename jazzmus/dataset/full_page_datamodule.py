"""
DataModule for full-page jazz leadsheet images.

Key differences from system-level:
- Loads full-page images (not cropped to individual systems)
- Variable image sizes (dynamic padding)
- Supports curriculum learning (variable # of systems)
"""

import cv2
import gin
import numpy as np
import torch
from lightning import LightningDataModule
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rich.progress import track

from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary, load_kern
from jazzmus.dataset.tokenizer import process_text


def load_full_page_set(
    dataset_dir,
    split="train",
    reduce_ratio=1.0,
    fixed_img_height=None,
    max_fix_img_width=None,
):
    """Load full-page images and ground truth from directory."""
    images_dir = Path(dataset_dir) / split / "images"
    gt_dir = Path(dataset_dir) / split / "ground_truth"

    if not images_dir.exists() or not gt_dir.exists():
        print(f"Warning: {split} split directories not found")
        return [], [], []

    # Get all image files
    img_files = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(img_files)} images in {split} split")

    x = []  # Images
    y = []  # Ground truth
    paths = []

    for img_file in track(img_files, description=f"Loading {split}"):
        # Load image
        img_path = str(img_file)
        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_raw is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        img = np.array(img_raw)

        # Resize image
        if fixed_img_height is not None:
            width = int(np.ceil(img.shape[1] * fixed_img_height / img.shape[0]))
            height = fixed_img_height
            if max_fix_img_width is not None:
                width = min(width, max_fix_img_width)
        else:
            width = int(np.ceil(img.shape[1] * reduce_ratio))
            height = int(np.ceil(img.shape[0] * reduce_ratio))

        img = cv2.resize(img, (width, height))

        # Load ground truth
        gt_file = gt_dir / img_file.stem
        if gt_file.with_suffix('.txt').exists():
            gt_content = load_kern(gt_file.with_suffix('.txt'))
        else:
            print(f"Warning: No ground truth for {img_file.name}")
            continue

        x.append(img)
        y.append(gt_content)
        paths.append(img_path)

    return x, y, paths


def batch_preparation_full_page(data):
    """Prepare batch with dynamic padding for variable-size full-page images."""
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]
    paths = [sample[3] for sample in data]

    # Find max dimensions in batch
    max_image_width = max(1000, max([img.shape[2] if len(img.shape) == 3 else img.shape[1] for img in images]))
    max_image_height = max(256, max([img.shape[1] if len(img.shape) == 3 else img.shape[0] for img in images]))

    # Pad images to max dimensions
    X_train = torch.ones(
        size=[len(images), 1, max_image_height, max_image_width],
        dtype=torch.float32
    )

    for i, img in enumerate(images):
        if len(img.shape) == 3:
            _, h, w = img.size()
        else:
            h, w = img.shape
        X_train[i, :, :h, :w] = img if torch.is_tensor(img) else torch.from_numpy(img).unsqueeze(0)

    # Prepare sequences
    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in), max_length_seq])
    y = torch.zeros(size=[len(gt), max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0 : len(seq) - 1] = torch.from_numpy(
            np.asarray([char for char in seq[:-1]])
        )

    for i, seq in enumerate(gt):
        y[i, 0 : len(seq) - 1] = torch.from_numpy(
            np.asarray([char for char in seq[1:]])
        )

    return X_train, decoder_input.long(), y.long(), paths


@gin.configurable
class FullPageGrandStaff(Dataset):
    """Dataset for full-page jazz leadsheet images."""

    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.path = None
        self.augment = augment
        self.w2i = None
        self.i2w = None
        self.padding_token = None

        super().__init__()

    def apply_teacher_forcing(self, sequence):
        """Apply random errors during teacher forcing."""
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if (
                np.random.rand() < self.teacher_forcing_error_rate
                and sequence[token] != self.padding_token
            ):
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        return errored_sequence

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def __getitem__(self, index):
        if self.x is None or len(self.x) == 0:
            raise ValueError("Dataset not initialized")

        x = self.x[index]
        y = self.y[index]
        path = self.path[index]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y, path

    def get_max_hw(self):
        """Get maximum height and width."""
        if self.x is None or len(self.x) == 0:
            return 256, 1000
        m_width = np.max([img.shape[1] if len(img.shape) == 2 else img.shape[2] for img in self.x])
        m_height = np.max([img.shape[0] if len(img.shape) == 2 else img.shape[1] for img in self.x])
        return m_height, m_width

    def get_max_seqlen(self):
        """Get maximum sequence length."""
        if self.y is None or len(self.y) == 0:
            return 1000
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        """Get vocabulary size."""
        return len(self.w2i) if self.w2i else 0

    def get_gt(self):
        """Get ground truth sequences."""
        return self.y

    def set_dictionaries(self, w2i, i2w):
        """Set word-to-index and index-to-word mappings."""
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i.get("<pad>", 0)

    def get_dictionaries(self):
        """Get word-to-index and index-to-word mappings."""
        return self.w2i, self.i2w

    def preprocess_gt(self, Y):
        """Preprocess ground truth by tokenizing."""
        for idx, krn in enumerate(Y):
            Y[idx] = (
                ["<bos>"] + process_text(lines=krn, tokenizer_type="word") + ["<eos>"]
            )
        return Y


@gin.configurable
class FullPageDataModule(LightningDataModule):
    """Lightning DataModule for full-page jazz leadsheet dataset."""

    def __init__(
        self,
        data_path="data/handwritten",
        vocab_name="full_page_vocab",
        batch_size=1,
        num_workers=4,
        fixed_img_height=256,
        max_fix_img_width=None,
    ) -> None:
        super().__init__()

        console_print = print  # Use regular print for clarity
        console_print("Initializing FullPageDataModule with parameters:")
        console_print(f"\tData path: {data_path}")
        console_print(f"\tVocab name: {vocab_name}")
        console_print(f"\tBatch size: {batch_size}")
        console_print(f"\tFixed image height: {fixed_img_height}")
        console_print(f"\tMax image width: {max_fix_img_width}")

        self.data_path = data_path
        self.vocab_name = vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fixed_img_height = fixed_img_height
        self.max_fix_img_width = max_fix_img_width

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage="fit"):
        """Setup train/val/test datasets."""
        console_print = print

        # Load datasets
        train_x, train_y, train_paths = load_full_page_set(
            self.data_path,
            split="train",
            fixed_img_height=self.fixed_img_height,
            max_fix_img_width=self.max_fix_img_width,
        )

        val_x, val_y, val_paths = load_full_page_set(
            self.data_path,
            split="val",
            fixed_img_height=self.fixed_img_height,
            max_fix_img_width=self.max_fix_img_width,
        )

        test_x, test_y, test_paths = load_full_page_set(
            self.data_path,
            split="test",
            fixed_img_height=self.fixed_img_height,
            max_fix_img_width=self.max_fix_img_width,
        )

        # Create dataset objects
        self.train_set = FullPageGrandStaff(augment=True)
        self.val_set = FullPageGrandStaff(augment=False)
        self.test_set = FullPageGrandStaff(augment=False)

        # Set data
        self.train_set.x = train_x
        self.train_set.y = self.train_set.preprocess_gt(train_y.copy())
        self.train_set.path = train_paths

        self.val_set.x = val_x
        self.val_set.y = self.val_set.preprocess_gt(val_y.copy())
        self.val_set.path = val_paths

        self.test_set.x = test_x
        self.test_set.y = self.test_set.preprocess_gt(test_y.copy())
        self.test_set.path = test_paths

        # Build vocabulary
        console_print(f"\nBuilding vocabulary from {len(train_y)} training samples...")
        w2i, i2w = check_and_retrieveVocabulary(
            [self.train_set.y, self.val_set.y, self.test_set.y],
            "vocab",
            self.vocab_name,
        )

        # Set vocabularies
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

        console_print(f"✓ Vocabulary size: {len(w2i)}")
        console_print(f"✓ Train set: {len(self.train_set)} images")
        console_print(f"✓ Val set: {len(self.val_set)} images")
        console_print(f"✓ Test set: {len(self.test_set)} images")

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=batch_preparation_full_page,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_full_page,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_full_page,
        )
