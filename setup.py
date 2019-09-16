from setuptools import setup

setup(name='osmuf',
      version='0.2',
      install_requires=[
            "seaborn",
      ],
      description='Urban Form analysis from OpenStreetMap',
      url='http://github.com/atelierlibre/osmuf',
      author='AtelierLibre',
      author_email='mail@atelierlibre.org',
      license='MIT',
      packages=['osmuf'],
      zip_safe=False)
