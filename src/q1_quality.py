"""Q1.1–1.2 main: domain directions, original comparison and conflict corrections."""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args, _ = parser.parse_known_args()
    # Entropy/semantic options apply only to the second experiment.
    domain_args = ["--output-dir", str(args.output_dir)] if args.output_dir else []
    subprocess.run([sys.executable, str(root/"q1_domain_direction.py"), *domain_args], check=True)
    subprocess.run([sys.executable, str(root/"q1_semantic_conflict.py"), *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
