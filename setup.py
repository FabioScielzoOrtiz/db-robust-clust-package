from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="db-robust-clust",
    version="0.1.5",
    author="Fabio Scielzo Ortiz",
    author_email="fabio.scielzoortiz@gmail.com",
    description="Apply distance based robust clustering for mixed data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabioScielzoOrtiz/db_robust_clust",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['polars','numpy<=1.26.4', 'PyDistances', 'pandas', 'scikit-learn-extra', 'tqdm', 'setuptools', 'pyarrow', 'matplotlib', 'seaborn'],
    python_requires=">=3.7"
)