"""Entry point for running as module: python -m ultra_light_memory"""

from .server import run_server


def main():
    run_server()


if __name__ == "__main__":
    main()
