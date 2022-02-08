from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Latent model package (NER, JNER)'
LONG_DESCRIPTION = 'Python package that contains tools for latent representation project, including pipeline and machine learning tools.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="latent", 
        version=VERSION,
        author="Yuni Teh",
        author_email="<yuni@u.northwestern.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)