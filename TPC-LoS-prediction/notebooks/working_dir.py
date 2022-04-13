import os
from pathlib import Path


def set_working_dir():
    wd = os.getcwd()
    if wd.endswith("notebooks"):
        os.chdir(Path(wd).parent)
    print(f"Current working directory: {os.getcwd()}")


def test():
    wd_before = os.getcwd()
    set_working_dir()
    print(f"before: {wd_before}, after: {os.getcwd()}")


if __name__ == '__main__':
    test()
