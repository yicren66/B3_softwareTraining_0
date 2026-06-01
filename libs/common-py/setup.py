from setuptools import setup, find_packages

setup(
    name="jujube-common",
    version="0.1.0",
    description="Shared Python library for the Jujube agricultural AI platform",
    author="Jujube Team",
    packages=find_packages(include=["common", "common.*"]),
    install_requires=[
        "pydantic>=2.0",
        "pyjwt>=2.8",
        "structlog>=23.0",
        "asyncpg>=0.28",
        "redis>=5.0",
        "prometheus-client>=0.18",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
