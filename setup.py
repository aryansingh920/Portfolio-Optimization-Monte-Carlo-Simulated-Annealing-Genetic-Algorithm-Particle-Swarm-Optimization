# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()
                    and not line.startswith("#")]

setup(
    name="portfolio_optimization",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive portfolio optimization package using various algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/portfolio-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "optimize-portfolio=portfolio_optimization.main:main",
        ],
    },
)
