"""Standalone versioned UV inference."""
import argparse
from engine import UVEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    UVEngine().infer(args.input, args.output)

if __name__ == '__main__':
    main()
