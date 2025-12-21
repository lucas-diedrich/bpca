# bpca

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/bpca/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/bpca

Bayesian Principal Component Analysis

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install bpca:

<!--
1) Install the latest release of `bpca` from [PyPI][]:

```bash
pip install bpca
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lucas-diedrich/bpca.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

This package implements the algorithm proposed by Bishope, 1998. **Please cite the original authors**

> Bishop, C. Bayesian PCA. in Advances in Neural Information Processing Systems vol. 11 (MIT Press, 1998).

If you find this implementation useful, consider giving it a star on GitHub and cite this implementation



[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/bpca/issues
[tests]: https://github.com/lucas-diedrich/bpca/actions/workflows/test.yaml
[documentation]: https://bpca.readthedocs.io
[changelog]: https://github.com/lucas-diedrich/bpca/releases
[api documentation]: https://bpca.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/bpca
