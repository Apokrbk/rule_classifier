import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rule_induction_classifier",
    version="1.0",
    author="Damian Portasiński",
    author_email="damian.portasinski@gmail.com",
    description="Classifier based on induction of decision rules, that uses Roaring Bitmap for storing data.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)