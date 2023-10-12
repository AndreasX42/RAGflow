from setuptools import setup, find_packages

setup(
    name="my_package",  # Name of your package
    version="0.1",  # Version number
    description="A simple example package",  # Short description
    long_description=open("README.md").read(),  # Long description, usually the README
    long_description_content_type="text/markdown",  # If using a markdown README
    author="Your Name",  # Your name
    author_email="youremail@example.com",  # Your email
    url="https://github.com/yourusername/my_package",  # Project homepage
    packages=find_packages(),  # Automatically discover and include all packages in the package directory
    install_requires=[
        "requests",  # For example, if your project depends on the requests library
        "another_dependency>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
