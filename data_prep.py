"""
Data preparation script for JazzMus full-page dataset

This script:
1. Downloads the PRAIG/JAZZMUS dataset from HuggingFace
2. Extracts full-page images and kern ground truth
3. Removes bounding box data and XML files
4. Organizes data into images/ and ground_truth/ folders
"""

import json
from pathlib import Path
from datasets import load_dataset
from rich.progress import track
from rich.console import Console

console = Console()

# Directories
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "full_page"


def clean_sample(sample):
    """
    Extract image and full-page **kern transcription

    Args:
        sample: Single dataset sample (may be dict or JSON string)

    Returns:
        tuple: (image, kern_text)
    """
    # Handle JSON string format
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except:
            return None, ""

    image = None
    kern = ""

    # Extract image
    if 'image' in sample:
        image = sample['image']

    # Extract full-page kern from encodings
    if isinstance(sample, dict) and 'encodings' in sample:
        if '**kern' in sample['encodings']:
            kern = sample['encodings']['**kern']

    return image, kern


def process_dataset():
    """Download and process the PRAIG/JAZZMUS dataset"""
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold cyan]JazzMus Full-Page Data Preparation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")

    # Load dataset
    console.print("\n[bold blue]Loading PRAIG/JAZZMUS dataset...[/bold blue]")
    try:
        dataset = load_dataset("PRAIG/JAZZMUS", trust_remote_code=True)
        console.print(f"  [green]✓[/green] Dataset loaded successfully")
        console.print(f"  Available splits: {list(dataset.keys())}")
    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        return

    # Process each split
    stats = {
        'total_samples': 0,
        'total_chars': 0,
        'splits': {}
    }

    for split_name in dataset.keys():
        console.print(f"\n[bold blue]Processing {split_name} split...[/bold blue]")

        split_data = dataset[split_name]

        # Create output directories
        images_dir = OUTPUT_DIR / split_name / "images"
        gt_dir = OUTPUT_DIR / split_name / "ground_truth"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        split_stats = {'count': 0, 'chars': 0, 'errors': 0}

        # Process each sample
        for idx, sample in enumerate(track(split_data, description=f"  {split_name}")):
            try:
                # Extract image and kern
                image, kern = clean_sample(sample)

                if image is None or not kern:
                    split_stats['errors'] += 1
                    continue

                # Save image
                img_path = images_dir / f"{split_name}_{idx:04d}.jpg"
                image.save(img_path)

                # Save kern ground truth
                gt_path = gt_dir / f"{split_name}_{idx:04d}.txt"
                with open(gt_path, 'w', encoding='utf-8') as f:
                    f.write(kern)

                split_stats['count'] += 1
                split_stats['chars'] += len(kern)

            except Exception as e:
                console.print(f"  [yellow]Warning[/yellow]: Error processing sample {idx}: {e}")
                split_stats['errors'] += 1

        stats['splits'][split_name] = split_stats
        stats['total_samples'] += split_stats['count']
        stats['total_chars'] += split_stats['chars']

        console.print(f"  [green]✓[/green] Processed {split_stats['count']} samples")
        if split_stats['errors'] > 0:
            console.print(f"  [yellow]⚠[/yellow] {split_stats['errors']} errors")

    # Save metadata
    console.print("\n[bold blue]Creating metadata...[/bold blue]")
    metadata = {
        'dataset': 'PRAIG/JAZZMUS',
        'format': 'full-page kern',
        'removed': ['musicxml', 'bounding_boxes', 'systems'],
        'statistics': stats
    }

    metadata_path = OUTPUT_DIR / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"  [green]✓[/green] Metadata saved to: {metadata_path}")

    # Print summary
    console.print("\n[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold green]Data Preparation Complete![/bold green]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print(f"\n[bold]Output location:[/bold] {OUTPUT_DIR}")
    console.print(f"[bold]Total samples:[/bold] {stats['total_samples']}")
    console.print(f"[bold]Total kern characters:[/bold] {stats['total_chars']:,}")

    console.print("\n[bold]Per-split breakdown:[/bold]")
    for split_name, split_stats in stats['splits'].items():
        console.print(f"  {split_name}:")
        console.print(f"    - Images: {split_stats['count']}")
        console.print(f"    - Ground truth: {split_stats['count']}")
        console.print(f"    - Location: {OUTPUT_DIR / split_name}")

    console.print("\n[bold]Directory structure:[/bold]")
    console.print(f"  {OUTPUT_DIR}/")
    for split_name in stats['splits'].keys():
        console.print(f"    {split_name}/")
        console.print(f"      images/       # {stats['splits'][split_name]['count']} .jpg files")
        console.print(f"      ground_truth/ # {stats['splits'][split_name]['count']} .txt files")


if __name__ == "__main__":
    process_dataset()
