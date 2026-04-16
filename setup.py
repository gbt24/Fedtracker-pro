"""FedTracker-Pro 安装脚本。"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"

long_description = README_PATH.read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]


setup(
    name="fedtracker-pro",
    version="0.1.0",
    author="Research Team",
    description="Enhanced federated learning model protection framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src*", "experiments*"]),
    python_requires=">=3.10",
    install_requires=requirements,
)
