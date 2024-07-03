from setuptools import setup, find_packages # type: ignore

setup(
    name='lightestimation',
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        '': ['py.typed'],
    },
    include_package_data=True
)
