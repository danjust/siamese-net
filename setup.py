from setuptools import setup, find_packages
setup(
    name="siamnet",
    version="0.0.1",
    packages=['siamnet'],
    package_dir={'siamnet': 'src/siamnet'},
    package_data={'siamnet': [
            'data_omniglot/*.zip']},
)
