"""Setup script."""

import setuptools

setuptools.setup(
    name='bipmo',
    version='0.1.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'matplotlib',
        'numpy',
        'pandas',
        'pyomo',
        'scipy',
    ]
)
