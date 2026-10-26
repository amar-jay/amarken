"""Local preparation, training, calibration and prediction commands."""

import argparse

def parser():
    root = argparse.ArgumentParser(description="amarken")
    # ...
    return root


def main():
    args = parser().parse_args()
    #...