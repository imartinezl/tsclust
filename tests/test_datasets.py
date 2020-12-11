#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tsclust.datasets

def test_univariate_download():
    ds = tsclust.datasets.UCR_UEA_datasets()
    dataset_names = ds.list_univariate_datasets()
    dataset_name = dataset_names[0]
    X_train, y_train, X_test, y_test = ds.load_dataset(dataset_name)

    assert dataset_name in ds.list_cached_datasets()

def test_univariate_shape():
    ds = tsclust.datasets.UCR_UEA_datasets()
    dataset_names = ds.list_univariate_datasets()
    dataset_name = dataset_names[0]
    X_train, y_train, X_test, y_test = ds.load_dataset(dataset_name)

    assert X_train.ndim == 3
    assert X_test.ndim == 3
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[2] == 1
    assert X_test.shape[2] == 1

def test_multivariate_download():
    ds = tsclust.datasets.UCR_UEA_datasets()
    dataset_names = ds.list_multivariate_datasets()
    dataset_name = dataset_names[0]
    X_train, y_train, X_test, y_test = ds.load_dataset(dataset_name)

    assert dataset_name in ds.list_cached_datasets()

def test_multivariate_shape():
    ds = tsclust.datasets.UCR_UEA_datasets()
    dataset_names = ds.list_multivariate_datasets()
    dataset_name = dataset_names[0]
    X_train, y_train, X_test, y_test = ds.load_dataset(dataset_name)

    assert X_train.ndim == 3
    assert X_test.ndim == 3
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[2] > 1
    assert X_test.shape[2] > 1

def test_dataset_joined():
    ds = tsclust.datasets.UCR_UEA_datasets()
    dataset_names = ds.list_datasets()
    dataset_name = dataset_names[-1]
    X_train, y_train, X_test, y_test = ds.load_dataset(dataset_name)

    data_all = np.vstack([X_train, X_test])
    data = [x[~np.isnan(x).any(axis=1)] for x in data_all]
    labels = np.concatenate([y_train, y_test])
    labels_unique = np.unique(labels)
    label = labels_unique[0]

    X = [data[i] for i in range(len(data)) if labels[i] == label]
    N = len(X)
