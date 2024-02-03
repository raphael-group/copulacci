Motivation
----------

Recovering true correlation coefficient in the context of single-cell or spatial RNA-seq data can be a challenging task due to the sparse nature of genomic measurements. Spearman and Pearson measures are long standing go-to methods for estimating correlation coefficients for any two vectors of numbers. However for single-cell transcriptomics and more recently for spatial transcriptomics we have more information of the underlying dataset, for example, we know that the underlying data is always integers, moreover each cell/spot expresses a fixed number of total UMI counts that is distributed over the genes that are expressed. These two observation can be incorporated in a better estimation of the correlation coefficient.

Similation
-----------
To view this in action we can demonstrate the copula in effect via simulation `notebook/tutorial/Copula_based_correlation_coefficient`

Installation
------------
```python=3.9
conda create -n copulacci python=3.9
conda activate copulacci
git clone git@github.com:raphael-group/copulacci.git
cd copulacci
pip install .
```



