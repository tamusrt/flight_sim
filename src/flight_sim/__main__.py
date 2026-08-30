"""Core FS runner"""

import argparse


def main():
    """Dummy code to test arg parsing"""
    parser = argparse.ArgumentParser(description="SRT FS")

    # Define expected command line arguments
    parser.add_argument("--test_input", default=None, help="Test input")

    args = parser.parse_args()

    print(f"Initializing FS with test input {args.test_input}")


if __name__ == "__main__":
    main()
