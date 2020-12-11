#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains code that was borrowed from tslearn.

BSD 2-Clause License

Copyright (c) 2017, Romain Tavenard
All rights reserved.
"""


import os
import numpy as np
from scipy.io import arff
import csv
import zipfile
import tempfile
import shutil
import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
try:
    from zipfile import BadZipFile as BadZipFile
except ImportError:
    from zipfile import BadZipfile as BadZipFile

from tsclust.utils import _load_arff_uea, _load_txt_uea

print(os.path.abspath("."))


def extract_from_zip_url(url, target_dir=None, verbose=False):
    """Download a zip file from its URL and unzip it.

    Parameters
    ----------
    url : string
        URL from which to download.
    target_dir : str or None (default: None)
        Directory to be used to extract unzipped downloaded files.
    verbose : bool (default: False)
        Whether to print information about the process (cached files used, ...)

    Returns
    -------
    str or None
        Directory in which the zip file has been extracted if the process was
        successful, None otherwise
    """
    fname = os.path.basename(url)
    tmpdir = tempfile.mkdtemp()
    local_zip_fname = os.path.join(tmpdir, fname)
    urlretrieve(url, local_zip_fname)
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        zipfile.ZipFile(local_zip_fname, "r").extractall(path=target_dir)
        shutil.rmtree(tmpdir)
        if verbose:
            print(
                "Successfully extracted file %s to path %s"
                % (local_zip_fname, target_dir)
            )
        return target_dir
    except BadZipFile:
        shutil.rmtree(tmpdir)
        if verbose:
            sys.stderr.write("Corrupted zip file encountered, aborting.\n")
        return None


class UCR_UEA_datasets:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        base_dir = os.path.expanduser(
            os.path.join("~", ".tsclust", "datasets", "UCR_UEA")
        )
        self._data_dir = base_dir
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)
        try:
            url_univariate = (
                "http://www.timeseriesclassification.com/"
                + "Downloads/Archives/summaryUnivariate.csv"
            )
            self._list_univariate_filename = os.path.join(
                self._data_dir, os.path.basename(url_univariate)
            )
            urlretrieve(url_univariate, self._list_univariate_filename)

            url_multivariate = (
                "http://www.timeseriesclassification.com/"
                + "Downloads/Archives/summaryMultivariate.csv"
            )
            self._list_multivariate_filename = os.path.join(
                self._data_dir, os.path.basename(url_multivariate)
            )
            urlretrieve(url_multivariate, self._list_multivariate_filename)
        except:
            self._list_univariate_filename = None
            self._list_multivariate_filename = None
        self._ignore_list = ["Data Descriptions"]

    def list_univariate_datasets(self):
        datasets = []
        with open(self._list_univariate_filename) as f:
            for row in csv.DictReader(f):
                datasets.append(row["problem"])
        return datasets

    def list_multivariate_datasets(self):
        datasets = []
        with open(self._list_multivariate_filename) as f:
            for row in csv.DictReader(f):
                datasets.append(row["Problem"])
        return datasets

    def list_datasets(self):
        return self.list_univariate_datasets() + self.list_multivariate_datasets()

    def list_cached_datasets(self):
        return [
            path
            for path in os.listdir(self._data_dir)
            if os.path.isdir(os.path.join(self._data_dir, path))
            and path not in self._ignore_list
        ]

    def _has_files(self, dataset_name, ext=None):
        if ext is None:
            return self._has_files(dataset_name, ext="txt") or self._has_files(
                dataset_name, ext="arff"
            )
        else:
            full_path = os.path.join(self._data_dir, dataset_name)
            basename = os.path.join(full_path, dataset_name)
            return os.path.exists(basename + "_TRAIN.%s" % ext) and os.path.exists(
                basename + "_TEST.%s" % ext
            )

    def load_dataset(self, dataset_name):
        full_path = os.path.join(self._data_dir, dataset_name)

        if not self._has_files(dataset_name):
            url = (
                "http://www.timeseriesclassification.com/Downloads/%s.zip"
                % dataset_name
            )
            if os.path.isdir(full_path):
                for fname in os.listdir(full_path):
                    os.remove(os.path.join(full_path, fname))
            extract_from_zip_url(url, target_dir=full_path, verbose=False)
        if self._has_files(dataset_name, ext="txt"):
            X_train, y_train = _load_txt_uea(
                os.path.join(full_path, dataset_name + "_TRAIN.txt")
            )
            X_test, y_test = _load_txt_uea(
                os.path.join(full_path, dataset_name + "_TEST.txt")
            )
        elif self._has_files(dataset_name, ext="arff"):
            X_train, y_train = _load_arff_uea(
                os.path.join(full_path, dataset_name + "_TRAIN.arff")
            )
            X_test, y_test = _load_arff_uea(
                os.path.join(full_path, dataset_name + "_TEST.arff")
            )
        else:
            return None, None, None, None
        return X_train, y_train, X_test, y_test
