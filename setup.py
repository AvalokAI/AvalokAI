from setuptools import find_packages, setup

setup(
    name="avalokai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],
    author="Sankalp Garg",
    author_email="sankalp2621998@gmail.com",
    description="AvalokAI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AvalokAI/AvalokAI",  # Replace with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
