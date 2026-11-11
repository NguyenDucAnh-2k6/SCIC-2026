"""
EdNet Dataset Loader: StreamingZipReader
Safely streams CSV member files from EdNet KT1-KT4 archives without disk extraction.
"""

import zipfile
import io
import csv
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.data.ednet.loader")

class StreamingZipReader:
    """
    Safely streams CSV member files from either:
    1. A massive ZIP archive (such as EdNet KT1-KT4) without extracting to disk.
    2. An already extracted directory (e.g., on Kaggle: /kaggle/input/ednet-kt1-v2/KT1 or /kaggle/input/ednet-kt1-v2).
    """
    def __init__(self, zip_path: str = "data/EdNet-KT1.zip"):
        self.input_path = Path(zip_path)
        self.zip_path = self.input_path
        self._archive: Optional[zipfile.ZipFile] = None
        self._is_dir: bool = False
        self._root_dir: Optional[Path] = None

        self._resolve_source()

    def _resolve_source(self) -> None:
        """Determines whether the path is a ZIP archive or an extracted directory."""
        if self.input_path.is_file() and self.input_path.suffix.lower() == ".zip":
            self._is_dir = False
            return

        if self.input_path.is_dir():
            self._is_dir = True
            self._root_dir = self.input_path
            logger.info(f"Using extracted dataset directory: {self._root_dir}")
            return

        # Auto-fallback: Path might have been specified as '.../EdNet-KT1.zip' but exists as a directory
        stem_dir = self.input_path.parent / self.input_path.stem
        parent_dir = self.input_path.parent

        if stem_dir.is_dir():
            logger.info(f"ZIP file not found, but matching directory found at: {stem_dir}")
            self._is_dir = True
            self._root_dir = stem_dir
            return

        if (parent_dir / "KT1").is_dir():
            logger.info(f"Found 'KT1' folder in parent directory: {parent_dir / 'KT1'}")
            self._is_dir = True
            self._root_dir = parent_dir
            return

        self._is_dir = False

    def __enter__(self):
        if not self._is_dir:
            if not self.input_path.exists():
                raise FileNotFoundError(f"Neither ZIP archive nor directory found at: {self.input_path}")
            self._archive = zipfile.ZipFile(self.input_path, mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._archive:
            self._archive.close()
            self._archive = None

    def list_members(self, prefix: str = "KT1/", max_members: Optional[int] = None) -> List[str]:
        """Lists member paths (e.g., 'KT1/u246317.csv')."""
        if self._is_dir:
            target_search_dir = self._root_dir / prefix if prefix and (self._root_dir / prefix).is_dir() else self._root_dir
            csv_paths = list(target_search_dir.glob("**/*.csv"))
            members = []
            for p in csv_paths:
                try:
                    rel_path = p.relative_to(self._root_dir)
                    members.append(str(rel_path).replace("\\", "/"))
                except ValueError:
                    members.append(p.name)

            members = [m for m in members if not Path(m).name.startswith("._") and not Path(m).name.startswith(".")]
            members.sort()
            if max_members:
                members = members[:max_members]
            return members

        if not self._archive:
            raise RuntimeError("Reader is not open. Use with-statement context manager.")

        members = [
            m for m in self._archive.namelist()
            if m.startswith(prefix) and m.endswith(".csv") and not Path(m).name.startswith("._") and not Path(m).name.startswith(".")
        ]
        members.sort()
        if max_members:
            members = members[:max_members]
        return members

    def read_csv_member(self, member_path: str) -> List[Dict[str, str]]:
        """Reads CSV rows into a list of dicts."""
        if self._is_dir:
            real_path = self._root_dir / member_path
            if not real_path.exists():
                cand = list(self._root_dir.glob(f"**/{Path(member_path).name}"))
                if cand:
                    real_path = cand[0]
                else:
                    raise FileNotFoundError(f"Member {member_path} not found in {self._root_dir}")

            with open(real_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                return list(reader)

        if not self._archive:
            raise RuntimeError("Reader is not open. Use with-statement context manager.")

        try:
            with self._archive.open(member_path, mode="r") as f:
                text_stream = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")
                reader = csv.DictReader(text_stream)
                return list(reader)
        except KeyError:
            clean_name = Path(member_path).name
            candidates = [m for m in self._archive.namelist() if m.endswith(clean_name)]
            if candidates:
                with self._archive.open(candidates[0], mode="r") as f:
                    text_stream = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")
                    reader = csv.DictReader(text_stream)
                    return list(reader)
            raise FileNotFoundError(f"Member {member_path} not found in archive {self.input_path}")

