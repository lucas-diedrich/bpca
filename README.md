# bpca

[![Tests][badge-tests]][tests]
[![Integration Tests][badge-integration-tests]][integration-tests]
[![codecov](https://codecov.io/gh/lucas-diedrich/bpca/graph/badge.svg?token=SIA7YSWCET)](https://codecov.io/gh/lucas-diedrich/bpca)
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/bpca/test.yaml?branch=main
[badge-integration-tests]: https://github.com/lucas-diedrich/bpca/actions/workflows/integration-test.yaml/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/bpca

Bayesian Principal Component Analysis

## Getting started
BPCA follows the standard scikit-learn syntax

```python
from bpca import BPCA
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset["data"]

# Fit + Extract information
bpca = BPCA(n_components=2)
usage = bpca.fit_transform(X)
loadings = bpca.components_
explained_variance_ratio = bpca.explained_variance_ratio_
```

Please refer to the [documentation][], in particular, the [API documentation][].


## Installation

You need to have Python 3.11 or newer installed on your system.

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

See the [Release Notes][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

This package implements the algorithm proposed by Oba, 2003 and is built on the reference implementation by Stacklies et al, 2008 **Please cite the original authors**

> Oba, S. et al. A Bayesian missing value estimation method for gene expression profile data. Bioinformatics 19, 2088 - 2096 (2003).

> Stacklies, W., Redestig, H., Scholz, M., Walther, D. & Selbig, J. pcaMethods—a bioconductor package providing PCA methods for incomplete data. Bioinformatics 23, 1164 - 1167 (2007).

Generative model proposed by Bishop, 1998:
> Bishop, C. Bayesian PCA. in Advances in Neural Information Processing Systems vol. 11 (MIT Press, 1998).

If you find this implementation useful, consider giving it a star on GitHub and [cite this implementation](https://github.com/lucas-diedrich/bpca/blob/main/CITATION.cff)

> Diedrich, L. bpca [Computer software]. https://github.com/lucas-diedrich/bpca.git

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/bpca/issues
[tests]: https://github.com/lucas-diedrich/bpca/actions/workflows/test.yaml
[integration-tests]: https://github.com/lucas-diedrich/bpca/actions/workflows/integration-test.yaml
[documentation]: https://bpca.readthedocs.io
[Release Notes]: https://github.com/lucas-diedrich/bpca/releases
[api documentation]: https://bpca.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/bpca
