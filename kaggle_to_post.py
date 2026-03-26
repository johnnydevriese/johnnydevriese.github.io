import os
import subprocess
import argparse
import yaml
import datetime
import re
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def get_kernel_date(slug):
    # Split slug into user and kernel name
    parts = slug.split('/')
    if len(parts) != 2:
        return datetime.date.today().isoformat()
    
    username, kernel_name = parts
    # Use kaggle kernels list to find the specific kernel and get its last run time
    # We search specifically for the kernel name to limit results
    command = ["kaggle", "kernels", "list", "--user", username, "--search", kernel_name, "--csv"]
    output = run_command(command)
    
    if output:
        # Simple CSV parsing
        lines = output.strip().split('\n')
        if len(lines) > 1:
            # ref,title,author,lastRunTime,totalVotes
            # Find the line that matches our slug exactly in the first column
            for line in lines[1:]:
                cols = line.split(',')
                if cols[0] == slug:
                    # lastRunTime is usually at index 3
                    # Format is '2025-02-09 04:35:01'
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', cols[3])
                    if date_match:
                        return date_match.group(1)
    
    print(f"Warning: Could not fetch original date for {slug}. Using today's date.")
    return datetime.date.today().isoformat()

def fetch_and_convert(slug):
    # Create a temp directory for pulling
    temp_dir = Path("temp_kaggle")
    temp_dir.mkdir(exist_ok=True)
    
    # Fetch the original date
    original_date = get_kernel_date(slug)
    print(f"Original date for {slug}: {original_date}")
    
    # Pull the kernel
    print(f"Pulling kernel: {slug}")
    run_command(["kaggle", "kernels", "pull", "-p", str(temp_dir), slug])
    
    # Find the .ipynb file
    ipynb_files = list(temp_dir.glob("*.ipynb"))
    if not ipynb_files:
        print("No .ipynb file found.")
        return
    
    ipynb_path = ipynb_files[0]
    kernel_name = ipynb_path.stem
    
    # Convert to markdown
    print(f"Converting {ipynb_path} to markdown")
    run_command(["jupyter", "nbconvert", "--to", "markdown", str(ipynb_path)])
    
    md_path = ipynb_path.with_suffix(".md")
    if not md_path.exists():
        print(f"Markdown file {md_path} not found.")
        return
    
    # Process the markdown file
    with open(md_path, "r") as f:
        content = f.read()
    
    # Extract images and fix paths
    image_dir_src = temp_dir / f"{kernel_name}_files"
    target_image_dir = Path(f"public/assets/images/posts/{kernel_name}")
    
    if image_dir_src.exists():
        target_image_dir.mkdir(parents=True, exist_ok=True)
        for img in image_dir_src.glob("*"):
            shutil.copy(img, target_image_dir / img.name)
        
        # Update image paths in markdown
        # nbconvert uses ![png](kernel_name_files/kernel_name_X_Y.png)
        rel_image_path = f"/assets/images/posts/{kernel_name}/"
        content = content.replace(f"{kernel_name}_files/", rel_image_path)

    # Remove broken attachment: references (embedded notebook images that can't be recovered)
    # These come from Jupyter cell attachments that nbconvert doesn't extract
    content = re.sub(r'!\[[^\]]*\]\(attachment:[^\)]+\)', '*[Image not available - embedded notebook attachment]*', content)

    # Prepare Frontmatter
    title = kernel_name.replace("-", " ").title()
    
    frontmatter = {
        "layout": "post",
        "title": title,
        "date": original_date,
        "categories": "kaggle machine-learning",
        "slug": kernel_name
    }
    
    # Prepend frontmatter
    fm_str = "---\n" + yaml.dump(frontmatter) + "---\n\n"
    final_content = fm_str + content
    
    # Output to content/posts
    post_filename = f"{original_date}-{kernel_name}.md"
    post_path = Path("content/posts") / post_filename
    os.makedirs("content/posts", exist_ok=True)
    
    with open(post_path, "w") as f:
        f.write(final_content)
    
    print(f"Successfully created post: {post_path}")
    
    # Clean up
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and convert Kaggle kernel to blog post")
    parser.add_argument("--slug", required=True, help="The Kaggle kernel slug (e.g., user/kernel-name)")
    args = parser.parse_args()
    
    fetch_and_convert(args.slug)
