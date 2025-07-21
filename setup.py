from setuptools import setup, find_packages

setup(
    name='llm-zipster',
    version='0.1.0',
    description='LLM arithmetic coding compressor/decompressor',
    author='Krishna C. Garikipati',
    url='https://github.com/gkchai/llm-zipster',
    py_modules=['llm_zipster'],
    install_requires=[
        'torch',
        'transformers',
        'tqdm',
        'numpy',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'llm-zipster=llm_zipster:main',
        ],
    },
) 