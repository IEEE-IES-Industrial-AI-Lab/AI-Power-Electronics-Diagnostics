from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-power-electronics-diagnostics",
    version="0.1.0",
    author="IEEE IES Industrial AI Lab",
    description="AI-based fault detection for power electronics: inverter and motor drive diagnostics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IEEE-IES-Industrial-AI-Lab/AI-Power-Electronics-Diagnostics",
    packages=find_packages(exclude=["notebooks", "benchmarks", "datasets"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "ped-train=training.train:main",
            "ped-evaluate=training.evaluate:main",
            "ped-benchmark=benchmarks.benchmark_all_models:main",
        ]
    },
)
