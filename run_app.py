#!/usr/bin/env python3
"""
Quick launcher for the Caixa Enginyers Streamlit App
"""

import subprocess
import sys
import argparse
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description="Launch the Streamlit app")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the scored municipalities CSV file"
    )
    args = parser.parse_args()
    
    app_path = Path(__file__).parent / "app data" / "streamlit_app_scored.py"
    
    if not app_path.exists():
        print(f"App not found at {app_path}")
        sys.exit(1)
    
    print("Launching Caixa Enginyers Branch Location Optimizer...")
    print(f"App: {app_path}")
    if args.data_path:
        print(f"Data: {args.data_path}")
    print("\nThe app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Ensure Streamlit's file watcher is disabled at the environment level
        # This protects against hosts that ignore CLI flags or use a different runner.
        os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

        cmd = [
            "streamlit", "run", str(app_path),
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Add data path argument if provided
        if args.data_path:
            cmd.extend(["--", "--data-path", args.data_path])
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nApp stopped")
    except FileNotFoundError:
        print("\nStreamlit not found. Install with:")
        print("   pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
