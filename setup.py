import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PLSBI",
    version="0.0.1",
    author="Xuan Kan",
    author_email="xuan.kan@emory.edu",
    description="PLSBI is a unified, modular and reproducible package established for brain network analysis with Partial Least Square Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wayfear/PLSBI",
    project_urls={
        "Bug Tracker": "https://github.com/Wayfear/PLSBI/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
