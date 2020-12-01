import numpy as np


def add_one(number):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series using a distance metric defined by
    the user and return both the path and the similarity.

    Similarity is computed as the cumulative cost along the aligned time
    series.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.

    Valid values for metric are the same as for scikit-learn
    `pairwise_distances`_ function i.e. a string (e.g. "euclidean",
    "sqeuclidean", "hamming") or a function that is used to compute the
    pairwise distances. See `scikit`_ and `scipy`_ documentations for more
    information about the available metrics.

    Parameters
    ----------
    s1 : array, shape = (sz1, d) if metric!="precomputed", (sz1, sz2) otherwise
        A time series or an array of pairwise distances between samples.

    s2 : array, shape = (sz2, d), optional (default: None)
        A second time series, only allowed if metric != "precomputed".

    metric : string or callable (default: "euclidean")
        Function used to compute the pairwise distances between each points of
        `s1` and `s2`.

        If metric is "precomputed", `s1` is assumed to be a distance matrix.

        If metric is an other string, it must be one of the options compatible
        with sklearn.metrics.pairwise_distances.

        Alternatively, if metric is a callable function, it is called on pairs
        of rows of `s1` and `s2`. The callable should take two 1 dimensional
        arrays as input and return a value indicating the distance between
        them.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    **kwds
        Additional arguments to pass to sklearn pairwise_distances to compute
        the pairwise distances.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score (sum of metric along the wrapped time series).

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    #>>> dtw_path_from_metric(s1, s2, metric="sqeuclidean")  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by defining a custom distance function:

    # >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    # >>> dtw_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by using a precomputed distance matrix as input:

    # >>> from sklearn.metrics.pairwise import pairwise_distances
    # >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    # >>> dtw_path_from_metric(dist_matrix,
    # ...                      metric="precomputed")  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Notes
    --------
    By using a squared euclidean distance metric as shown above, the output
    path is the same as the one obtained by using dtw_path but the similarity
    score is the sum of squared distances instead of the euclidean distance.

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """
    return number + 1


def create_array(n):
    return np.array([n])
