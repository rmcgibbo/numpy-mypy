from setuptools import setup, find_packages


setup(
    name='numpy-mypy',
    packages=find_packages(),
    include_package_data=True,
    entry_points = {
        'mypy.plugins': [
            'numpy = numpy_plugin.plugin:plugin'
        ]
    },
)