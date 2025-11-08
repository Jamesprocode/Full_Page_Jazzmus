"""
Data preparation script for JazzMus full-page dataset with piece-level splits

This script:
1. Downloads the PRAIG/JAZZMUS dataset from HuggingFace
2. Extracts piece titles from MusicXML to identify duplicates/versions
3. Groups images by piece title
4. Splits unique pieces 70/10/20
5. Puts ALL versions of training pieces into train set
6. Extracts FULL-PAGE images and kern ground truth
7. Removes bounding box data and XML files
8. Organizes data into images/ and ground_truth/ folders for curriculum training

Structure:
- Each sample has 'image' (full page) and 'annotation' (JSON with systems/encodings)
- annotation contains:
  - systems: array of individual staff systems with bounding boxes
  - encodings: full-page **kern transcription + **musicxml (for title extraction)
"""

import json
import ast
import re
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from rich.progress import track
from rich.console import Console

console = Console()

# Directories
DATA_DIR = Path(__file__).parent / "data"


def extract_title_from_musicxml(musicxml_str):
    """Extract title from MusicXML string."""
    if not musicxml_str:
        return None
    match = re.search(r'<movement-title>(.*?)</movement-title>', musicxml_str)
    if match:
        return match.group(1).strip()
    match = re.search(r'<work-title>(.*?)</work-title>', musicxml_str)
    if match:
        return match.group(1).strip()
    return None


def extract_composer_from_musicxml(musicxml_str):
    """Extract composer from MusicXML string."""
    if not musicxml_str:
        return None
    match = re.search(r'<creator type="composer">(.*?)</creator>', musicxml_str)
    if match:
        return match.group(1).strip()
    return None


def extract_piece_id(annotation_data, idx):
    """
    Extract piece ID from annotation (title - composer).

    Args:
        annotation_data: Annotation (string or dict)
        idx: Image index for debugging

    Returns:
        str: Piece ID or fallback ID
    """
    # Parse annotation if it's a string
    if isinstance(annotation_data, str):
        try:
            annotation = json.loads(annotation_data)
        except json.JSONDecodeError:
            try:
                annotation = ast.literal_eval(annotation_data)
            except:
                return f"Unknown_{idx}"
    else:
        annotation = annotation_data

    # Extract title and composer from MusicXML
    if isinstance(annotation, dict) and 'encodings' in annotation:
        encodings = annotation['encodings']
        if isinstance(encodings, dict) and 'musicxml' in encodings:
            musicxml = encodings['musicxml']
            title = extract_title_from_musicxml(musicxml)
            composer = extract_composer_from_musicxml(musicxml)

            if title:
                return f"{title} - {composer}" if composer else title

    # Fallback
    return f"Unknown_{idx}"


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


def process_dataset(dataset_name="PRAIG/JAZZMUS", output_name="full_page", max_images=None, split_ratio=(0.7, 0.1, 0.2), random_seed=42, synthetic_only=False):
    """
    Download and process a JazzMus dataset from HuggingFace

    Args:
        dataset_name: HuggingFace dataset name (default: "PRAIG/JAZZMUS")
        output_name: Output folder name (default: "full_page")
        max_images: Limit number of images (for testing)
        split_ratio: (train, val, test) ratio
        random_seed: Random seed for reproducibility
        synthetic_only: If True, save all data to a single 'synthetic' folder (no split)
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

    # Extract split ratios
    train_pct, val_pct, test_pct = split_ratio

    # Handle synthetic data (no splitting needed)
    if synthetic_only:
        console.print(f"\n[bold blue]Processing synthetic data (no split)...[/bold blue]")

        splits = {
            'synthetic': np.arange(len(images_list))
        }

        console.print(f"  Total synthetic samples: {len(images_list)}")

    else:
        # Pass 1: Extract piece titles and group by title (identify versions/duplicates)
        console.print(f"\n[bold blue]Pass 1: Extracting piece titles from MusicXML...[/bold blue]")

        piece_titles = {}  # idx -> "Title - Composer"
        title_to_indices = defaultdict(list)  # "Title - Composer" -> [idx1, idx2, ...]

        for idx in track(range(len(images_list)), description="Extracting titles"):
            orig_idx, _ = images_list[idx]
            sample = dataset[orig_idx]
            annotation = sample['annotation']

            piece_id = extract_piece_id(annotation, orig_idx)
            piece_titles[idx] = piece_id
            title_to_indices[piece_id].append(idx)

        unique_pieces = len(title_to_indices)
        total_versions = len(images_list)
        avg_versions = total_versions / unique_pieces if unique_pieces > 0 else 0

        console.print(f"  [green]✓[/green] Found {unique_pieces} unique pieces")
        console.print(f"  [green]✓[/green] Total images (including versions): {total_versions}")
        console.print(f"  [green]✓[/green] Average versions per piece: {avg_versions:.2f}")

        duplicates = {title: indices for title, indices in title_to_indices.items() if len(indices) > 1}
        if duplicates:
            console.print(f"  [yellow]⚠[/yellow] Pieces with duplicates: {len(duplicates)}")
            console.print(f"  [yellow]⚠[/yellow] Total duplicate scores: {sum(len(indices) for indices in duplicates.values())}")

        # Pass 2: Split unique pieces (not individual images)
        console.print(f"\n[bold blue]Pass 2: Splitting unique piece titles...[/bold blue]")

        unique_titles = list(title_to_indices.keys())
        np.random.seed(random_seed)
        shuffled_titles = np.array(unique_titles)
        np.random.shuffle(shuffled_titles)

        train_pct, val_pct, test_pct = split_ratio
        n_pieces = len(shuffled_titles)
        n_test = int(n_pieces * test_pct)
        n_val = int(n_pieces * val_pct)
        n_train = n_pieces - n_test - n_val

        test_titles = shuffled_titles[:n_test].tolist()
        val_titles = shuffled_titles[n_test:n_test + n_val].tolist()
        train_titles = shuffled_titles[n_test + n_val:].tolist()

        # Convert titles to indices - only first occurrence goes to val/test
        val_indices = [title_to_indices[title][0] for title in val_titles]
        test_indices = [title_to_indices[title][0] for title in test_titles]

        assigned_to_val_or_test = set(val_indices + test_indices)

        # Train: ALL indices not assigned to val/test (includes all versions of training pieces)
        train_indices = [idx for idx in range(len(images_list)) if idx not in assigned_to_val_or_test]

        splits = {
            'train': np.array(train_indices),
            'val': np.array(val_indices),
            'test': np.array(test_indices)
        }

        console.print(f"  Unique pieces split:")
        console.print(f"    Train: {n_train} pieces ({train_pct*100:.0f}%)")
        console.print(f"    Val:   {n_val} pieces ({val_pct*100:.0f}%)")
        console.print(f"    Test:  {n_test} pieces ({test_pct*100:.0f}%)")
        console.print(f"\n  Total images (including all versions):")
        console.print(f"    Train: {len(train_indices)} samples (includes all versions)")
        console.print(f"    Val:   {len(val_indices)} samples")
        console.print(f"    Test:  {len(test_indices)} samples")

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
    parser.add_argument("--synthetic", action="store_true",
                        help="Process as synthetic data (no split, all to single 'synthetic' folder)")

    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        max_images=args.max_images,
        split_ratio=(args.train_pct, args.val_pct, args.test_pct),
        random_seed=args.seed,
        synthetic_only=args.synthetic
    )
