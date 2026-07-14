import os
from pathlib import Path
from cognicore.extension.server import _get_project_id
import hashlib

def test_get_project_id_with_git(tmp_path, monkeypatch):
    # Create a mock git repo structure
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    
    sub_dir = repo_root / "src" / "deep" / "module"
    sub_dir.mkdir(parents=True)
    
    # Change current working directory to the deep module
    monkeypatch.chdir(sub_dir)
    
    project_id = _get_project_id()
    
    expected_path_str = str(repo_root.resolve())
    expected_hash = hashlib.sha256(expected_path_str.encode("utf-8")).hexdigest()
    
    assert project_id == expected_hash

def test_get_project_id_without_git(tmp_path, monkeypatch):
    # No .git anywhere
    some_dir = tmp_path / "no_repo" / "deep"
    some_dir.mkdir(parents=True)
    
    monkeypatch.chdir(some_dir)
    
    # Mock is_dir to always return False for .git so it doesn't find the real repository's .git
    original_is_dir = Path.is_dir
    def mock_is_dir(self):
        if self.name == ".git":
            return False
        return original_is_dir(self)
    
    monkeypatch.setattr(Path, "is_dir", mock_is_dir)
    
    project_id = _get_project_id()
    
    expected_path_str = str(some_dir.resolve())
    expected_hash = hashlib.sha256(expected_path_str.encode("utf-8")).hexdigest()
    
    assert project_id == expected_hash
