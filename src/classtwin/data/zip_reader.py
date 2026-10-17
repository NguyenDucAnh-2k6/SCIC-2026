import zipfile
import io
import csv
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict
from ..utils.logging import get_logger

logger = get_logger("classtwin.data.zip_reader")

class StreamingZipReader:
    """
    Safely streams CSV member files from a massive ZIP archive (such as EdNet KT1-KT4)
    without extracting the archive to disk or blowing up memory.
    """
    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.zip_path}")
        self._archive: Optional[zipfile.ZipFile] = None

    def open(self) -> "StreamingZipReader":
        self._archive = zipfile.ZipFile(self.zip_path, "r")
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
        """Returns sorted list of valid member paths in the ZIP archive."""
        if not self._archive:
            raise RuntimeError("Zip file is not opened. Use within a context manager.")
        members = [
            name for name in self._archive.namelist()
            if not name.endswith("/") and name.startswith(prefix) and not name.startswith("__MACOSX")
        ]
        members.sort()
        if max_members is not None:
            members = members[:max_members]
        return members

    def read_csv_member(self, member_name: str) -> List[Dict[str, str]]:
        """Reads a single CSV member file and yields dict rows."""
        if not self._archive:
            raise RuntimeError("Zip file is not opened.")
        with self._archive.open(member_name, "r") as raw_f:
            text_stream = io.TextIOWrapper(raw_f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text_stream)
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
        logger.info(f"Streaming {len(members)} members from {self.zip_path.name}")
        for member in members:
            try:
                rows = self.read_csv_member(member)
                yield member, rows
            except Exception as e:
                logger.warning(f"Error reading member {member}: {e}")
                continue
