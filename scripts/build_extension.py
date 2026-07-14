import os
import zipfile
import fnmatch
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent.parent
EXTENSION_DIR = ROOT_DIR / "extension"
BUILD_DIR = ROOT_DIR / "build"
BUNDLE_NAME = "cognicore-memory.mcpb"

def get_ignored_patterns():
    ignore_file = EXTENSION_DIR / ".mcpbignore"
    patterns = []
    if ignore_file.exists():
        with open(ignore_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    # Always ignore build dir
    patterns.append("build/")
    patterns.append(f"{BUNDLE_NAME}")
    return patterns

def is_ignored(path_str, patterns):
    for pattern in patterns:
        if pattern.endswith('/'):
            if pattern[:-1] in path_str.split('/'):
                return True
        elif fnmatch.fnmatch(path_str, pattern):
            return True
        elif fnmatch.fnmatch(os.path.basename(path_str), pattern):
            return True
    return False

def build_bundle():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BUILD_DIR / BUNDLE_NAME
    
    patterns = get_ignored_patterns()
    print(f"Building {BUNDLE_NAME}...")
    
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add manifest and pyproject from extension dir
        zf.write(EXTENSION_DIR / "manifest.json", "manifest.json")
        zf.write(EXTENSION_DIR / "pyproject.toml", "pyproject.toml")
        if (EXTENSION_DIR / "README.md").exists():
            zf.write(EXTENSION_DIR / "README.md", "README.md")
            
        # Add cognicore package
        cognicore_dir = ROOT_DIR / "cognicore"
        for root, dirs, files in os.walk(cognicore_dir):
            rel_root = os.path.relpath(root, ROOT_DIR)
            
            # Filter ignored dirs in place to stop traversing
            dirs[:] = [d for d in dirs if not is_ignored(f"{rel_root}/{d}".replace('\\', '/'), patterns)]
            
            for file in files:
                file_rel = f"{rel_root}/{file}".replace('\\', '/')
                if not is_ignored(file_rel, patterns):
                    file_path = os.path.join(root, file)
                    zf.write(file_path, file_rel)
                    
    print(f"Bundle created at: {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1024:.2f} KB")
    
if __name__ == "__main__":
    build_bundle()
