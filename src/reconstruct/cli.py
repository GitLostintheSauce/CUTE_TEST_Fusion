"""CLI entry point for CUTE equilibrium reconstruction."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="CUTE equilibrium reconstruction")
    parser.add_argument("--shot", required=True, help="Path to shot HDF5 file")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    args = parser.parse_args()
    print(f"Reconstruction not yet implemented. Shot: {args.shot}, Output: {args.output}")


if __name__ == "__main__":
    main()
