from setuptools import setup, find_packages


def read_requirements(file):
    try:
        with open(file, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []


setup(
    name="common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements_dev.txt"),
        "viz": read_requirements("requirements_viz.txt"),
        "backend": read_requirements("requirements_backend.txt"),
        "train": read_requirements("requirements_train.txt"),
    },
)
