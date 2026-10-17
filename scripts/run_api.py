import sys
from pathlib import Path
import uvicorn

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.inference.api import app
from src.classtwin.utils.logging import get_logger

logger = get_logger("scripts.run_api")

def main():
    logger.info("Starting ClassTwin Decision Support FastAPI service...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
