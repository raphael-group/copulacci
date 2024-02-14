"""
"""
import setuptools

setuptools.setup(
    name='copulacci',
    version='v0.0.1',
    python_requires='>=3.9',
    packages=['copulacci'],
    package_dir={'': 'src'},
    author='Hirak Sarkar',
    author_email='hirak@princeton.edu',
    description='A count-based model for delineating cell-cell interactions in spatial transcriptomics data',
    url='https://github.com/raphael-group/copulacci/',
    install_requires=[
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'tqdm',
        'scipy',
        'networkx',
        'jupyterlab',
        'joblib',
        'scanpy',
        'anndata',
        'omnipath',
        'squidpy',
        'spatialdm',
        'adjustText'
    ],
    include_package_data = True,
    package_data = {
        '' : ['*.txt']
        },
    license='BSD',
    platforms=["Linux", "MacOs"],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        'spatial transcriptomics',
        'neural network',
        'tissue layer'],
    entry_points={'console_scripts': 'copulacci=copulacci.__main__:main'}
)
