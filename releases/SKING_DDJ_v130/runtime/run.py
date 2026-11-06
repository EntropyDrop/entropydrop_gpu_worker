"""Standalone SKING_DDJ_v130 inference for smoke checks and integrations."""
import argparse
from engine import UVEngine

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();UVEngine().infer(a.input,a.output)

if __name__=='__main__':main()
