from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyagents",
    version="0.0.1",
    author="Chinellato Diego & Campardo Giorgia",
    author_email="chine.diego@gmail.com",
    description="Implementations of DRL algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diegochine/pyagents",
    project_urls={
        "Bug Tracker": "https://github.com/diegochine/deep-reinforcement-learning/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('.'),
    install_requires=[
        'numpy',
        'tensorflow>=2.10',
        'tensorflow-probability',
        'gymnasium',
        'tqdm',
        'gin-config',
        'h5py',
        'wandb',
    ],
    python_requires=">=3.7",
)
