from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import shutil
import sys

R_INSTALL_SCRIPT = (
    "options(repos=c(techtonique='https://r-packages.techtonique.net', "
    "CRAN='https://cloud.r-project.org')); "
    "if (!requireNamespace('ahead', quietly=TRUE)) "
    "install.packages('ahead', dependencies=TRUE)"
)


def install_r_ahead():
    if shutil.which("Rscript") is None:
        sys.exit(
            "\nERROR: R was not found on this system, but the 'ahead' Python "
            "package requires it.\nInstall R first: https://cran.r-project.org "
            "then re-run `pip install ahead`.\n"
        )
    try:
        subprocess.run(["Rscript", "-e", R_INSTALL_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"\nERROR: failed to install the R 'ahead' package ({e}).\n"
            "Try running this manually in R:\n"
            f"  {R_INSTALL_SCRIPT}\n"
        )


class BuildPyCommand(build_py):
    def run(self):
        install_r_ahead()
        build_py.run(self)


class DevelopCommand(develop):
    def run(self):
        install_r_ahead()
        develop.run(self)


__version__ = "0.38.7"

with open("requirements.txt", encoding="utf-8") as f:
    all_reqs = f.read().split("\n")
install_requires = [x.strip() for x in all_reqs if x.strip() and "git+" not in x]

setup(
    name="ahead",
    version=__version__,
    description="Time series forecasting with Machine Learning and uncertainty quantification",
    long_description="A package for time series forecasting with Machine Learning and uncertainty quantification",
    license="BSD3 Clause Clear",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author="T. Moudiki",
    author_email="thierry.moudiki@gmail.com",
    install_requires=install_requires,
    python_requires=">=3.8",
    cmdclass={"build_py": BuildPyCommand, "develop": DevelopCommand},
)
