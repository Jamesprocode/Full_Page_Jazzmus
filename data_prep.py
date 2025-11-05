"""
Data preparation script for JazzMus full-page dataset

This script:
1. Downloads the PRAIG/JAZZMUS dataset from HuggingFace
2. Extracts FULL-PAGE images and kern ground truth
3. Removes bounding box data and XML files
4. Organizes data into images/ and ground_truth/ folders for curriculum training

Structure:
- Each sample has 'image' (full page) and 'annotation' (JSON with systems)
- annotation contains:
  - systems: array of individual staff systems with bounding boxes
  - encodings: full-page **kern transcription (THIS IS WHAT WE WANT!)
"""

import json
import ast
from pathlib import Path
from datasets import load_dataset
from rich.progress import track
from rich.console import Console

console = Console()

# Directories
DATA_DIR = Path(__file__).parent / "data"


def extract_full_page_data(image, annotation_data, idx):
    """
    Extract full-page image and full-page **kern transcription

    Args:
        image: PIL Image of the full page
        annotation_data: Annotation (string or dict) containing encodings
        idx: Image index for debugging

    Returns:
        tuple: (image, full_page_kern)
    """
    # Parse annotation if it's a string
    if isinstance(annotation_data, str):
        try:
            annotation = json.loads(annotation_data)
        except json.JSONDecodeError:
            try:
                annotation = ast.literal_eval(annotation_data)
            except Exception as e:
                console.print(f"[yellow]Warning[/yellow]: Failed to parse annotation {idx}: {e}")
                return None, None
    else:
        annotation = annotation_data

    # Extract full-page kern from encodings
    full_page_kern = ""
    if isinstance(annotation, dict) and 'encodings' in annotation:
        encodings = annotation['encodings']
        if isinstance(encodings, dict) and '**kern' in encodings:
            full_page_kern = encodings['**kern']

    if not full_page_kern:
        return None, None

    return image, full_page_kern


def process_dataset(dataset_name="PRAIG/JAZZMUS", output_name="full_page", max_images=None, split_ratio=(0.7, 0.1, 0.2), random_seed=42):
    """
    Download and process a JazzMus dataset from HuggingFace

    Args:
        dataset_name: HuggingFace dataset name (default: "PRAIG/JAZZMUS")
        output_name: Output folder name (default: "full_page")
        max_images: Limit number of images (for testing)
        split_ratio: (train, val, test) ratio
        random_seed: Random seed for reproducibility
    """
    import numpy as np

    OUTPUT_DIR = DATA_DIR / output_name

    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print(f"[bold cyan]JazzMus Full-Page Data Preparation: {dataset_name}[/bold cyan]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")

    # Load dataset
    console.print(f"\n[bold blue]Loading {dataset_name} dataset...[/bold blue]")
    try:
        dataset = load_dataset(dataset_name, split="train", num_proc=4, trust_remote_code=True)
        console.print(f"  [green]✓[/green] Dataset loaded: {len(dataset)} images")
    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        return

    # Debug first sample
    first_sample = dataset[0]
    console.print(f"\n[bold yellow]DEBUG: First sample structure[/bold yellow]")
    console.print(f"  Keys: {list(first_sample.keys())}")
    if 'annotation' in first_sample:
        ann_data = first_sample['annotation']
        if isinstance(ann_data, str):
            try:
                parsed = json.loads(ann_data)
                console.print(f"  Annotation keys: {list(parsed.keys())}")
                if 'encodings' in parsed:
                    console.print(f"  Encodings keys: {list(parsed['encodings'].keys())}")
            except:
                pass

    # Limit dataset size if specified
    num_images = min(len(dataset), max_images) if max_images else len(dataset)

    # Create output directories
    console.print(f"\n[bold blue]Processing {num_images} full-page images...[/bold blue]")

    # Process all images
    images_list = []
    kerns_list = []
    failed = 0

    for idx in track(range(num_images), description="Processing"):
        sample = dataset[idx]
        image = sample['image']
        annotation = sample['annotation']

        # Extract full-page data
        full_image, full_kern = extract_full_page_data(image, annotation, idx)

        if full_image is None or full_kern is None:
            failed += 1
            continue

        images_list.append((idx, full_image))
        kerns_list.append((idx, full_kern))

    console.print(f"\n  [green]✓[/green] Successfully extracted {len(images_list)} full-page samples")
    if failed > 0:
        console.print(f"  [yellow]⚠[/yellow] {failed} samples failed")

    # Split into train/val/test
    console.print(f"\n[bold blue]Creating train/val/test splits...[/bold blue]")

    n_total = len(images_list)
    indices = np.arange(n_total)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_pct, val_pct, test_pct = split_ratio
    n_test = int(n_total * test_pct)
    n_val = int(n_total * val_pct)
    n_train = n_total - n_test - n_val

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    console.print(f"  Train: {n_train} samples ({train_pct*100:.0f}%)")
    console.print(f"  Val:   {n_val} samples ({val_pct*100:.0f}%)")
    console.print(f"  Test:  {n_test} samples ({test_pct*100:.0f}%)")

    # Save data for each split
    console.print(f"\n[bold blue]Saving data...[/bold blue]")

    stats = {
        'total_samples': 0,
        'total_chars': 0,
        'splits': {}
    }

    for split_name, split_indices in splits.items():
        # Create directories
        images_dir = OUTPUT_DIR / split_name / "images"
        gt_dir = OUTPUT_DIR / split_name / "ground_truth"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        split_stats = {'count': 0, 'chars': 0}

        # Save each sample in the split
        for i, data_idx in enumerate(split_indices):
            orig_idx, image = images_list[data_idx]
            _, kern = kerns_list[data_idx]

            # Save image
            img_path = images_dir / f"{split_name}_{i:04d}.jpg"
            image.convert("RGB").save(img_path, "JPEG")

            # Save kern
            gt_path = gt_dir / f"{split_name}_{i:04d}.txt"
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(kern)

            split_stats['count'] += 1
            split_stats['chars'] += len(kern)

        stats['splits'][split_name] = split_stats
        stats['total_samples'] += split_stats['count']
        stats['total_chars'] += split_stats['chars']

        console.print(f"  [green]✓[/green] {split_name}: {split_stats['count']} samples saved")

    # Save metadata
    console.print(f"\n[bold blue]Creating metadata...[/bold blue]")
    metadata = {
        'dataset': dataset_name,
        'output_name': output_name,
        'format': 'full-page kern',
        'removed': ['musicxml', 'bounding_boxes', 'individual_systems'],
        'split_ratio': {
            'train': train_pct,
            'val': val_pct,
            'test': test_pct
        },
        'random_seed': random_seed,
        'statistics': stats
    }

    metadata_path = OUTPUT_DIR / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"  [green]✓[/green] Metadata saved to: {metadata_path}")

    # Final summary
    console.print("\n" + "[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold green]Data Preparation Complete![/bold green]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print(f"\n[bold]Output location:[/bold] {OUTPUT_DIR}")
    console.print(f"[bold]Total samples:[/bold] {stats['total_samples']}")
    console.print(f"[bold]Total kern characters:[/bold] {stats['total_chars']:,}")

    console.print("\n[bold]Directory structure:[/bold]")
    console.print(f"  {OUTPUT_DIR}/")
    for split_name in ['train', 'val', 'test']:
        count = stats['splits'][split_name]['count']
        console.print(f"    {split_name}/")
        console.print(f"      images/       # {count} .jpg files")
        console.print(f"      ground_truth/ # {count} .txt files")

    console.print("\n[bold green]Ready for curriculum training![/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare JazzMus full-page dataset")
    parser.add_argument("--dataset_name", type=str, default="PRAIG/JAZZMUS",
                        help="HuggingFace dataset name (default: PRAIG/JAZZMUS)")
    parser.add_argument("--output_name", type=str, default="full_page",
                        help="Output folder name (default: full_page)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for testing)")
    parser.add_argument("--train_pct", type=float, default=0.7,
                        help="Training set percentage")
    parser.add_argument("--val_pct", type=float, default=0.1,
                        help="Validation set percentage")
    parser.add_argument("--test_pct", type=float, default=0.2,
                        help="Test set percentage")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        max_images=args.max_images,
        split_ratio=(args.train_pct, args.val_pct, args.test_pct),
        random_seed=args.seed
    )
