import sys
import os
import time
import argparse
import subprocess
import threading
import webbrowser
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import uvicorn
from src.classtwin.utils.logging import get_logger

logger = get_logger("scripts.run_api")

def ensure_frontend_build(frontend_dir: Path):
    """Tự động kiểm tra và build frontend React nếu chưa có thư mục dist."""
    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        logger.info("Chưa tìm thấy bản build React (frontend/dist). Đang tự động chạy 'npm run build'...")
        try:
            # Kiểm tra node_modules
            if not (frontend_dir / "node_modules").exists():
                logger.info("Đang cài đặt npm packages cho frontend...")
                subprocess.run("npm install", cwd=str(frontend_dir), shell=True, check=True)

            subprocess.run("npm run build", cwd=str(frontend_dir), shell=True, check=True)
            logger.info("Build React frontend hoàn tất thành công!")
        except Exception as e:
            logger.warning(f"Không thể build tự động frontend: {e}. Hệ thống sẽ sử dụng giao diện fallback.")

def main():
    parser = argparse.ArgumentParser(description="ClassTwin Digital Twin & Decision Support Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng FastAPI backend (mặc định: 8000)")
    parser.add_argument("--dev", action="store_true", help="Chạy đồng thời Vite Dev Server với Hot-Reload ở cổng 3000")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()

    frontend_dir = repo_root / "frontend"
    dev_process = None

    if args.dev:
        # Chế độ Dev: Chạy đồng thời npm run dev
        logger.info("Đang khởi chạy Vite React Dev Server (port 3000) song song...")
        try:
            dev_process = subprocess.Popen("npm run dev", cwd=str(frontend_dir), shell=True)
        except Exception as e:
            logger.error(f"Lỗi khởi chạy Vite dev server: {e}")

        target_url = "http://localhost:3000/"
    else:
        # Chế độ Production / Standalone: Tự động đảm bảo đã build React dist
        ensure_frontend_build(frontend_dir)
        target_url = f"http://localhost:{args.port}/"

    # Tự động mở trình duyệt sau khi server backend và UI đã sẵn sàng 100%
    if not args.no_browser:
        def wait_and_open_browser():
            import socket
            logger.info("Đang kiểm tra kết nối để đảm bảo backend và UI sẵn sàng...")
            # Kiểm tra port backend
            for _ in range(30):
                time.sleep(0.4)
                try:
                    with socket.create_connection(("localhost", args.port), timeout=0.5):
                        break
                except (ConnectionRefusedError, OSError):
                    continue

            # Nếu chế độ dev, kiểm tra thêm cổng 3000 của Vite
            if args.dev:
                for _ in range(30):
                    try:
                        with socket.create_connection(("localhost", 3000), timeout=0.5):
                            break
                    except (ConnectionRefusedError, OSError):
                        time.sleep(0.3)

            time.sleep(0.4)
            logger.info(f"Hệ thống đã sẵn sàng! Đang mở giao diện ClassTwin: {target_url}")
            webbrowser.open(target_url)

        threading.Thread(target=wait_and_open_browser, daemon=True).start()

    logger.info(f"ClassTwin Server đang khởi động tại: {target_url}")
    try:
        from src.classtwin.inference.api import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if dev_process:
            logger.info("Đang dọn dẹp và tắt tiến trình Vite Dev Server...")
            dev_process.terminate()

if __name__ == "__main__":
    main()
