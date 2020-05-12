import setuptools

setuptools.setup(
    name="digipyro",
    version="0.0",
    packages=setuptools.find_packages(),
    author="The DIYnamics Team",
    author_email="DIYnamicsTeam@gmail.com",
    description="Digitally Rotate a movie in Python",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python",
        "tkinter",
    ],
    scripts=["digipyro/scripts/.py"],
    license="Apache",
    keywords="education",
    url="https://github.com/DIYnamics/digipyro",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ]
)
