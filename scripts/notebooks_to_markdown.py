#!/usr/bin/env python3

"""
Python Examples Script

This script converts Jupyter notebooks from the LlamaIndex Python documentation
into markdown files for the Astro-based documentation site. It performs the following operations:

1. Finds all .ipynb files in the .build/python/docs/docs/examples directory
2. Converts each notebook to markdown using nbconvert
3. Transforms the first header into frontmatter format for Astro
4. Fixes static file path references
5. Extracts and saves embedded images from notebooks
6. Copies additional image files (PNG/JPG) from notebook directories
7. Transforms directory names to sentence case (replacing underscores with spaces)
8. Copies the converted files to the final destination at src/content/docs/python-examples
9. Copies _static and data directories for proper asset handling

The script uses parallel processing to efficiently convert multiple notebooks.
It relies on the repo having already been cloned by python-docs.js.
"""

from pathlib import Path
from nbconvert import MarkdownExporter
import nbformat
import base64
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from tqdm import tqdm
import argparse
from pathlib import Path

# Use tomllib for Python 3.11+, otherwise fallback to tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        print(
            "Error: 'tomli' package not found. Please install it (`pip install tomli`) for Python < 3.11."
        )
        exit(1)

def save_image_resources(resources, dest_dir, notebook_name):

    """Save any images from the notebook to the destination directory."""
    if not resources or 'outputs' not in resources:
        return
    
    # Create an images directory for this notebook
    #images_dir = dest_dir / f"{notebook_name}_files"
    images_dir = dest_dir # the notebooks don't know about the folder structure
    images_dir.mkdir(exist_ok=True)
    
    # Save each image
    for filename, data in resources['outputs'].items():
        if isinstance(data, str) and data.startswith('data:image'):
            # Extract the image data and format from data URI
            match = re.match(r'data:image/(\w+);base64,(.*)', data)
            if match:
                img_format, img_data = match.groups()
                img_path = images_dir / f"{filename}.{img_format}"
                
                # Decode and save the image
                with open(img_path, 'wb') as f:
                    f.write(base64.b64decode(img_data))
        elif isinstance(data, bytes):
            # Handle binary image data
            img_path = images_dir / filename
            
            # Save binary data directly
            with open(img_path, 'wb') as f:
                f.write(data)

def add_frontmatter(notebook_info: dict, md_content: str) -> str:
    """Convert the first header in markdown content to frontmatter.
    
    Args:
        md_content: The markdown content to process
        
    Returns:
        The modified markdown content with frontmatter
    """
    print("ASKED TO ADD FM: ", notebook_info)
    # Find the first header
    header_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
    if not header_match:
        return md_content
        
    title = header_match.group(1)
    
    # Create frontmatter
    frontmatter =  f"""\
---
layout: recipe
colab: {notebook_info.get("colab")}
toc: True
title: "{title}"
featured: {notebook_info.get("featured", False)}
experimental: {notebook_info.get("experimental", False)}
tags: {notebook_info.get("tags", [])}
---
"""
    # Remove the original header and add frontmatter
    return frontmatter + md_content[header_match.end():].lstrip()

def fix_static_paths(md_content: str) -> str:
    """Fix references to static files in markdown content.
    
    Args:
        md_content: The markdown content to process
        
    Returns:
        The modified markdown content with fixed static paths
    """
    # Replace any ../../../_static/ with ../_static/
    return re.sub(r'\.\./\.\./\.\./_static/', '../_static/', md_content)

def convert_single_notebook(notebook: dict, source_dir: Path, dest_dir: Path) -> Tuple[Path, bool, str]:
    """Convert a single notebook to markdown.
    
    Returns:
        Tuple[Path, bool, str]: (notebook_path, success, error_message)
    """
    
    notebook_path =  notebook.get("notebook_path")
    notebook_info = notebook.get("notebook_info")
    print("NOTEBOOK: ", notebook_path)
    print("INFO: ", notebook_info)
    try:
        # Calculate relative path from source directory
        rel_path = notebook_path.relative_to(source_dir)
        print(rel_path)
        # Create corresponding destination directory
        dest_subdir = dest_dir / rel_path.parent
        dest_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate destination markdown file path
        dest_file = dest_subdir / f"{rel_path.stem}.md"
        
        # Initialize the markdown exporter
        md_exporter = MarkdownExporter()
        
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            output = nbformat.read(f, as_version=4)
        
        # Convert to markdown
        (body, resources) = md_exporter.from_notebook_node(output)
        
        # Convert header to frontmatter
        body = add_frontmatter(notebook_info, body)
        
        # Fix static file paths
        body = fix_static_paths(body)
        
        # Save any images
        save_image_resources(resources, dest_subdir, rel_path.stem)
        
        # Write the markdown file
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write(body)
        
        return notebook_path, True, ""
    except Exception as e:
        return notebook_path, False, str(e)

def transform_dir_name(name: str) -> str:
    """Transform a directory name to sentence case and replace underscores with spaces.
    
    Args:
        name: The directory name to transform
        
    Returns:
        The transformed directory name
    """
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Convert to sentence case
    return name[0].upper() + name[1:].lower()

def copy_to_final_destination(source_dir: Path, dest_dir: Path):
    """Copy all converted files to the final destination directory.
    
    Args:
        source_dir: Source directory containing converted files
        dest_dir: Final destination directory
    """
    # Remove existing destination directory if it exists
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    
    # Create parent directories if they don't exist
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the entire directory tree with transformed names
    for item in source_dir.rglob("*"):
        if item.is_file():
            # Calculate the relative path
            rel_path = item.relative_to(source_dir)
            # Transform each directory name in the path, but not the filename
            new_parts = []
            for part in rel_path.parts[:-1]:  # All parts except the filename
                if part in ['_static', 'data']:
                    new_parts.append(part)
                else:
                    new_parts.append(transform_dir_name(part))
            # Add the original filename unchanged
            new_parts.append(rel_path.parts[-1])
            # Create the new path
            new_path = dest_dir.joinpath(*new_parts)
            # Create parent directories if they don't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy the file
            shutil.copy2(item, new_path)
    
    # Copy _static directory
    static_source = Path(".build/python/docs/docs/_static")
    static_dest = dest_dir / "_static"
    if static_source.exists():
        shutil.copytree(static_source, static_dest)
    
    # Copy data directory
    data_source = Path(".build/python/docs/docs/examples/data")
    data_dest = dest_dir / "data"
    if data_source.exists():
        shutil.copytree(data_source, data_dest)
    
    print(f"Copied converted files to {dest_dir}")

def copy_directory_images(source_dir: Path, dest_dir: Path):
    """Copy all PNG and JPG files from a source directory to the destination.
    
    Args:
        source_dir: Source directory to search for images
        dest_dir: Destination directory to copy images to
    """
    # Find all PNG and JPG files in the source directory
    image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    
    # Copy each image file
    for img_file in image_files:
        dest_file = dest_dir / img_file.name
        shutil.copy2(img_file, dest_file)

def convert_notebooks(index_data):
    # Source and destination directories
    source_dir = Path("notebooks/")
    temp_dir = Path(".tmp/")
    final_dir = Path("markdowns/")
    
    # Clean up temporary directory if it exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Create temporary destination directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .ipynb files recursively
    recipes = index_data.get("recipe", [])
    colab_base_url = index_data["config"]["colab"]
    notebooks = []
    for i, recipe_data in enumerate(recipes):
        notebook_info = {
            "file": Path(recipe_data["notebook"]),
            "title": recipe_data["title"],
            "colab": f"{colab_base_url.rstrip('/')}/{recipe_data["notebook"]}",
            "featured": recipe_data.get("featured", False),
            "experimental": recipe_data.get("experimental", False),
            "tags": recipe_data.get("tags", []),
            "relative_repo_path": recipe_data["notebook"],  # Pass relative path for image fixing
        }
        notebook_path = Path(recipe_data["notebook"])
        notebooks.append({
            "notebook_path": notebook_path,
            "notebook_info": notebook_info
        })
    notebook_paths = [notebook['notebook_path'] for notebook in notebooks ]
    total_notebooks = len(notebook_paths)

    print(f"Found {total_notebooks} notebooks to convert")
    
    # Use ThreadPoolExecutor for parallel processing
    # Using threads instead of processes because the work is I/O bound
    with ThreadPoolExecutor() as executor:
        # Submit all conversion tasks
        future_to_notebook = {
            executor.submit(convert_single_notebook, notebook, source_dir, temp_dir): notebook
            for notebook in notebooks
        }
        
        # Process completed tasks as they finish
        with tqdm(total=total_notebooks, desc="Converting notebooks") as pbar:
            for future in as_completed(future_to_notebook):
                notebook_path, success, error = future.result()
                pbar.update(1)
                
                if not success:
                    print(f"\nError converting {notebook_path}: {error}")
    
    # Copy images from each notebook directory
    for notebook_path in notebook_paths:
        source_subdir = notebook_path.parent
        rel_path = source_subdir.relative_to(source_dir)
        dest_subdir = temp_dir / rel_path
        copy_directory_images(source_subdir, dest_subdir)
    
    # Copy converted files to final destination
    copy_to_final_destination(temp_dir, final_dir)

def load_config(config_path):
    """Loads and parses the TOML configuration file."""
    print(f"Looking for index file at: {config_path}")
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return None

    try:
        index_toml_content = config_path.read_text(encoding="utf-8")
        index_data = tomllib.loads(index_toml_content)
        print("Successfully loaded configuration.")
        # Basic validation
        if "config" not in index_data or "colab" not in index_data["config"]:
            print(
                "Error: 'config' section with 'colab' URL base missing in configuration."
            )
            return None
        if "recipe" not in index_data:
            print("Warning: No recipes found in 'recipe' section of configuration.")
            # Allow continuing if recipes might be empty
            index_data["recipe"] = []
        return index_data
    except Exception as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Jupyter Notebooks to Markdown"
    )
    parser.add_argument(
        "--config",
        default="index.toml",
        help="Path to the TOML configuration file relative to project root (default: index.toml)",
    ) 
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path_arg = Path(args.config)
    if config_path_arg.is_absolute():
        config_file_path = config_path_arg
        root_path = script_dir.parent  # Guess project root based on script location
        print(f"Using absolute config path: {config_file_path}")
        print(f"Guessed Project root path: {root_path}")
    else:
        # Assume config path is relative to the project root,
        # and project root is parent of script's directory
        root_path = script_dir.parent
        config_file_path = (root_path / config_path_arg).resolve()
        print(f"Project root path: {root_path}")

    # Load configuration
    index_data = load_config(config_file_path)
    if index_data is None:
        exit(1)
    
    convert_notebooks(index_data)
