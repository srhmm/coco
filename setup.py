from setuptools import setup

setup(
    name='coco',
    version='1.0',
    packages=['coco', 'experiments', 'experiments.exp_coco', 'linc', 'linc.competitors', 'linc.competitors.dcdi_exp', 'linc.competitors.dcdi_exp.cam',
              'linc.competitors.dcdi_exp.dcdi', 'linc.competitors.dcdi_exp.dcdi.utils',
              'linc.competitors.dcdi_exp.dcdi.models', 'linc.competitors.dcdi_exp.gies', 'linc.sparse_shift',
              'linc.sparse_shift.datasets', 'linc.sparse_shift.datasets.tests', 'linc.sparse_shift.causal_learn',
              'vario', 'sparse_shift', 'sparse_shift.datasets', 'sparse_shift.datasets.tests',
              'sparse_shift.causal_learn'],
    url='',
    license='',
    author='anonymous',
    author_email='',
    description='coco'
)
