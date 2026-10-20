import zipfile
import io
import csv
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict
from ..utils.logging import get_logger

logger = get_logger("classtwin.data.zip_reader")

class StreamingZipReader:
    """
    Safely streams CSV member files from either:
    1. A massive ZIP archive (such as EdNet KT1-KT4) without extracting to disk.
    2. An already extracted directory (e.g., on Kaggle: /kaggle/input/ednet-kt1-v2/KT1 or /kaggle/input/ednet-kt1-v2).
    """
    def __init__(self, zip_path: str):
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

        if parent_dir.is_dir() and any(parent_dir.glob("*.csv")):
            logger.info(f"Parent directory contains CSV files directly: {parent_dir}")
            self._is_dir = True
            self._root_dir = parent_dir
            return

        raise FileNotFoundError(
            f"Archive file or extracted dataset folder not found: {self.input_path}. "
            "Please check the path in config."
        )

    def open(self) -> "StreamingZipReader":
        if not self._is_dir:
            self._archive = zipfile.ZipFile(self.input_path, "r")
        return self

    def close(self) -> None:
        if self._archive:
            self._archive.close()
            self._archive = None

    def __enter__(self) -> "StreamingZipReader":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_members(self, prefix: str = "", max_members: Optional[int] = None) -> List[str]:
        """Returns sorted list of valid member paths in the ZIP archive or folder."""
        if not self._is_dir:
            if not self._archive:
                raise RuntimeError("Zip file is not opened. Use within a context manager.")
            members = [
                name for name in self._archive.namelist()
                if not name.endswith("/") and name.startswith(prefix) and not name.startswith("__MACOSX")
            ]
        else:
            # Folder mode: find csv files
            csv_files = []
            # Check if there is a subfolder matching prefix like KT1/
            clean_prefix = prefix.strip("/")
            search_dir = self._root_dir
            if clean_prefix and (self._root_dir / clean_prefix).is_dir():
                target_folder = self._root_dir / clean_prefix
                csv_files = [
                    f"{clean_prefix}/{f.name}"
                    for f in target_folder.glob("*.csv")
                ]
            else:
                # Direct search inside _root_dir
                for f in search_dir.rglob("*.csv"):
                    rel = f.relative_to(search_dir).as_posix()
                    if not rel.startswith("__MACOSX"):
                        csv_files.append(rel)

            members = csv_files

        members.sort()
        if max_members is not None:
            members = members[:max_members]
        return members

    def read_csv_member(self, member_name: str) -> List[Dict[str, str]]:
        """Reads a single CSV member file and yields dict rows."""
        if not self._is_dir:
            if not self._archive:
                raise RuntimeError("Zip file is not opened.")
            with self._archive.open(member_name, "r") as raw_f:
                text_stream = io.TextIOWrapper(raw_f, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text_stream)
                return list(reader)
        else:
            # Folder mode
            file_path = self._root_dir / member_name
            if not file_path.exists():
                # Try finding by basename
                file_path = self._root_dir / Path(member_name).name
            if not file_path.exists() and (self._root_dir / "KT1").is_dir():
                file_path = self._root_dir / "KT1" / Path(member_name).name

            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found in directory: {member_name}")

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                return list(reader)

    def stream_members(
        self,
        prefix: str = "",
        max_members: Optional[int] = None
    ) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
        """
        Iterates over member CSVs yielding (member_name, rows).
        """
        members = self.list_members(prefix=prefix, max_members=max_members)
        source_name = self.input_path.name if not self._is_dir else str(self._root_dir)
        logger.info(f"Streaming {len(members)} members from {source_name}")
        for member in members:
            try:
                rows = self.read_csv_member(member)
                yield member, rows
            except Exception as e:
                logger.warning(f"Error reading member {member}: {e}")
                continue
