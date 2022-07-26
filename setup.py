
import setuptools



setuptools.setup(
    name="yamlFileReader",
    version="0.0.1",
    author="Akash Borgalli",
    author_email="akash.borgalli@gmail.com",
    description="A small package that would read params from yaml file and give as an dictionary",
    long_description="A small package that would read params from yaml file and give as an dictionary",
    long_description_content_type="text/markdown",
    url="", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",  #Specify which pyhton versions that you want to support
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


