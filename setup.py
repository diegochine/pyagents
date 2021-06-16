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
    python_requires=">=3.6",
)
