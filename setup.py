from setuptools import find_packages, setup

setup(
    name="avalokai",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    author="Sankalp Garg",
    author_email="sankalp2621998@gmail.com",
    description="DeepSearch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AvalokAI/DeepSearch",  # Replace with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
