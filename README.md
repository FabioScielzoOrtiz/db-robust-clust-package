# `db-robust-clust`

In the era of big data, data scientists are trying to solve real-world problems using multivariate
and heterogeneous datasets, i.e., datasets where for each unit multiple variables of different
nature are observed. Clustering may be a challenging problem when data are of mixed-type and
present an underlying correlation structure and outlying units.

In the paper ***Grané, A., Scielzo-Ortiz, F.: New distance-based clustering algorithms for large mixed-type data, Submitted to Journal of Classification (2025)***, new efficient robust clustering algorithms able to deal with large mixed-type data are developed and implemented in a **new Python package**, called `db-robust-clust`, hosted in the official Python Package Index (PyPI), the standard repository of packages for the Python programming language:: https://pypi.org/project/db_robust_clust/. 

Their performance is analyzed in rather complex mixed-type datasets,
both synthetic and real, where a wide variety of scenarios is considered regarding
size, the proportion of outlying units, the underlying correlation structure, and the
cluster pattern. The simulation study comprises four computational experiments
conducted on datasets of sizes ranging from 35k to 1M, in which the accuracy and
efficiency of the new proposals are tested and compared to those of existing clus-
tering alternatives. In addition, the goodness and computing time of the methods
under evaluation are tested on real datasets of varying sizes and patterns. MDS is
used to visualize clustering results.