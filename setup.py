from setuptools import setup


with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='gtsfutur',
    version='0.1.6.7.4',
    description='Simplify time series forecasting',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' ,
    license='MIT',
    packages=["GTSFutur"],
    author='Gregory Scafarto',
    author_email='gregory.scafarto@gmail.com',
    keywords=['Time series', 'forecasting','anomaly detection'],
    url='https://github.com/gregoritoo/GTSFutur',
    download_url=''
)

install_requires=[
          'tensorflow',
          'statsmodels',
          'sklearn',
          'matplotlib',
          'pandas',
          'PyWavelets',
          ]


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
