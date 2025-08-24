import setuptools

setuptools.setup(
    name='pydynet',
    version='1.0',
    description=
    'PyDyNet: Neuron Network (MLP, CNN, RNN, Transformer, ...) implementation using Numpy with Autodiff',
    author="Cun-Yuan Xing",
    author_email="xingcy@lamda.nju.edu.cn",
    maintainer="Cun-Yuan Xing",
    maintainer_email="xingcy@lamad.nju.edu.cn",
    packages=[
        'pydynet', 'pydynet/optim', 'pydynet/nn', 'pydynet/nn/modules',
        'pydynet/core'
    ],
    license='MIT License',
    install_requires=['numpy>=2.0.0'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/WeltXing/PyDyNet',
)
