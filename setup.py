from setuptools import setup, find_packages

setup(
    name='npview',
    version='0.1.0',
    description='Batch visualization of 2D numpy arrays and 1D slices',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'npview-examples=examples.plot_numpy_images:main',
        ],
    },
)
