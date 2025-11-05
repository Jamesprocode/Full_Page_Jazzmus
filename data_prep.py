"""
Data preparation script for JazzMus full-page dataset

This script:
1. Downloads the JazzMus dataset from HuggingFace
2. Removes bounding box data and XML files
3. Extracts only kern ground truth files
4. Organizes data for curriculum training
"""

import shutil
from pathlib import Path
import json

from datasets import load_dataset
from rich.progress import track
from rich.console import Console

console = Console()

# Directories
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FULL_PAGE_DIR = CLEAN_DIR / "full_page"
REGIONS_DIR = CLEAN_DIR / "regions"


def setup_directories():
    """Create necessary directories"""
    console.print("\n[bold blue]Setting up directories...[/bold blue]")

    for directory in [DATA_DIR, RAW_DIR, CLEAN_DIR, FULL_PAGE_DIR, REGIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"  Done {directory.name}")


def download_jazzmus_dataset():
    """Download JazzMus dataset from HuggingFace"""
    console.print("\n[bold blue]Downloading JazzMus dataset from HuggingFace...[/bold blue]")

    try:
        # Load the dataset
        console.print("  Loading dataset: [cyan]antoniorv6/JazzMus[/cyan]")
        dataset = load_dataset('antoniorv6/JazzMus', trust_remote_code=True)

        console.print("  [green]Success![/green] Dataset loaded successfully")
        console.print(f"  Available splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            console.print(f"    - {split_name}: {len(split_data)} samples")

        return dataset

    except Exception as e:
        console.print(f"  [red]Error[/red] loading dataset: {e}")
        return None


def clean_and_extract_kern(dataset, output_dir: Path):
    """
    Extract kern files from dataset and remove XML/bounding box data

    Args:
        dataset: HuggingFace dataset
        output_dir: Directory to save cleaned data
    """
    console.print("\n[bold blue]Cleaning and extracting kern files...[/bold blue]")

    # Create output directories
    images_dir = output_dir / "images"
    gt_dir = output_dir / "ground_truth"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_samples': 0,
        'kern_extracted': 0,
        'xml_removed': 0,
        'bbox_removed': 0,
    }

    # Process each split
    for split_name, split_data in dataset.items():
        console.print(f"\n  Processing split: [cyan]{split_name}[/cyan]")

        split_images_dir = images_dir / split_name
        split_gt_dir = gt_dir / split_name
        split_images_dir.mkdir(exist_ok=True)
        split_gt_dir.mkdir(exist_ok=True)

        for idx, sample in enumerate(track(split_data, description=f"  {split_name}")):
            stats['total_samples'] += 1

            # Extract image
            if 'image' in sample:
                image = sample['image']
                image_path = split_images_dir / f"{split_name}_{idx:04d}.jpg"
                image.save(image_path)

            # Extract kern transcription (ignore XML and bounding boxes)
            if 'kern' in sample or 'transcription' in sample:
                kern_content = sample.get('kern', sample.get('transcription', ''))

                # Save only kern file
                kern_path = split_gt_dir / f"{split_name}_{idx:04d}.txt"
                with open(kern_path, 'w', encoding='utf-8') as f:
                    f.write(kern_content)

                stats['kern_extracted'] += 1

            # Track removed data (just for stats)
            if 'xml' in sample or 'musicxml' in sample:
                stats['xml_removed'] += 1

            if 'bbox' in sample or 'bounding_box' in sample or 'boxes' in sample:
                stats['bbox_removed'] += 1

    # Print statistics
    console.print("\n[bold green]Extraction complete![/bold green]")
    console.print(f"  Total samples processed: {stats['total_samples']}")
    console.print(f"  Kern files extracted: {stats['kern_extracted']}")
    console.print(f"  XML data removed: {stats['xml_removed']}")
    console.print(f"  Bounding box data removed: {stats['bbox_removed']}")

    return stats


def organize_for_training(clean_dir: Path):
    """
    Organize cleaned data into training-ready structure

    Creates:
    - train/val/test splits
    - Proper directory structure for curriculum learning
    """
    console.print("\n[bold blue]Organizing data for training...[/bold blue]")

    images_dir = clean_dir / "images"
    gt_dir = clean_dir / "ground_truth"

    # Create split directories
    for split in ['train', 'val', 'test']:
        (clean_dir / 'organized' / split / 'images').mkdir(parents=True, exist_ok=True)
        (clean_dir / 'organized' / split / 'ground_truth').mkdir(parents=True, exist_ok=True)

    # If dataset has explicit splits, use them
    for split in ['train', 'val', 'test']:
        split_img_dir = images_dir / split
        split_gt_dir = gt_dir / split

        if split_img_dir.exists():
            console.print(f"  Organizing {split} split...")

            # Copy images
            for img_file in split_img_dir.glob("*.jpg"):
                dest = clean_dir / 'organized' / split / 'images' / img_file.name
                shutil.copy2(img_file, dest)

            # Copy ground truth
            for gt_file in split_gt_dir.glob("*.txt"):
                dest = clean_dir / 'organized' / split / 'ground_truth' / gt_file.name
                shutil.copy2(gt_file, dest)

            img_count = len(list((clean_dir / 'organized' / split / 'images').glob("*.jpg")))
            gt_count = len(list((clean_dir / 'organized' / split / 'ground_truth').glob("*.txt")))
            console.print(f"    Done {split}: {img_count} images, {gt_count} kern files")


def create_metadata(clean_dir: Path):
    """Create metadata file with dataset information"""
    console.print("\n[bold blue]Creating metadata...[/bold blue]")

    organized_dir = clean_dir / 'organized'
    metadata = {
        'dataset': 'JazzMus',
        'source': 'HuggingFace: antoniorv6/JazzMus',
        'format': 'kern',
        'splits': {}
    }

    for split in ['train', 'val', 'test']:
        split_dir = organized_dir / split
        if split_dir.exists():
            images = list((split_dir / 'images').glob("*.jpg"))
            gt_files = list((split_dir / 'ground_truth').glob("*.txt"))

            metadata['splits'][split] = {
                'num_samples': len(images),
                'num_gt_files': len(gt_files),
            }

    # Save metadata
    metadata_path = clean_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"  Metadata saved to: {metadata_path}")
    console.print("\n[bold]Dataset Summary:[/bold]")
    for split, info in metadata['splits'].items():
        console.print(f"  {split}: {info['num_samples']} samples")


def verify_data_integrity(clean_dir: Path):
    """Verify that all images have corresponding kern files"""
    console.print("\n[bold blue]Verifying data integrity...[/bold blue]")

    organized_dir = clean_dir / 'organized'
    issues = []

    for split in ['train', 'val', 'test']:
        split_dir = organized_dir / split
        if not split_dir.exists():
            continue

        images = {f.stem for f in (split_dir / 'images').glob("*.jpg")}
        gt_files = {f.stem for f in (split_dir / 'ground_truth').glob("*.txt")}

        # Check for missing ground truth
        missing_gt = images - gt_files
        if missing_gt:
            issues.append(f"{split}: {len(missing_gt)} images missing ground truth")

        # Check for extra ground truth
        extra_gt = gt_files - images
        if extra_gt:
            issues.append(f"{split}: {len(extra_gt)} extra ground truth files")

    if issues:
        console.print("[yellow]  Issues found:[/yellow]")
        for issue in issues:
            console.print(f"    - {issue}")
    else:
        console.print("[green]  All data verified successfully![/green]")


def main():
    """Main execution function"""
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold cyan]JazzMus Data Preparation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")

    # Step 1: Setup directories
    setup_directories()

    # Step 2: Download dataset
    dataset = download_jazzmus_dataset()

    if dataset is None:
        console.print("[red]Failed to download dataset. Exiting.[/red]")
        return

    # Step 3: Clean and extract kern files
    clean_and_extract_kern(dataset, CLEAN_DIR)

    # Step 4: Organize for training
    organize_for_training(CLEAN_DIR)

    # Step 5: Create metadata
    create_metadata(CLEAN_DIR)

    # Step 6: Verify integrity
    verify_data_integrity(CLEAN_DIR)

    # Final summary
    console.print("\n[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold green]Data preparation complete![/bold green]")
    console.print(f"[bold]Clean data location:[/bold] {CLEAN_DIR / 'organized'}")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")


if __name__ == "__main__":
    main()
