from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="anomaly-detection-framework",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Distributed Anomaly Detection Framework with Federated Learning and Causal Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research/anomaly-detection-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.9.1",
            "flake8==6.1.0",
            "mypy==1.5.1",
            "isort==5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adf-edge=edge.main:main",
            "adf-coordinator=coordinator.main:main",
            "adf-benchmark=experiments.benchmark:main",
        ],
    },
)
