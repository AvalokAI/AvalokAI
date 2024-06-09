from setuptools import find_packages, setup

setup(
    name="avalokai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],
    author="Avalok AI",
    author_email="",
    description="AvalokAI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AvalokAI/AvalokAI",  # Replace with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
