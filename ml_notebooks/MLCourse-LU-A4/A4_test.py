# Do NOT modify this file!

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as sk_scale
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.utils import shuffle as sk_shuffle

from A4 import *


def test_center_and_scale():
    # And uncenter_and_unscale...
    X = np.array([[0.0, 0.5, 7], [1.0, 1.5, -10], [2.0, 4.5, 0], [3.0, 5.5, 999]])
    X_norm, orig_center, orig_std = center_and_scale(X)
    assert helpful_eq(X_norm.mean(), 0)
    assert helpful_eq(X_norm.var(), 1)
    sk_X_norm = sk_scale(X)
    assert helpful_eq(X_norm, sk_X_norm)
    assert helpful_eq(orig_center, [1.5, 3.0, 249.])
    assert helpful_eq(orig_std, [1.118, 2.062, 433.055])
    X = np.dot(np.random.rand(2, 2), np.random.randn(2, 100)).T
    csX = center_and_scale(X)
    assert helpful_eq(csX[0], sk_scale(X))
    assert helpful_eq(uncenter_and_unscale(*csX), X)


def test_svd_to_components_and_score():
    for k in [1, 2]:
        X = sk_scale(np.array([[0.0, 0.4],
                               [1.0, 1.6],
                               [2.0, 2.4],
                               [3.0, 3.6]]))
        U, S, V_transpose = np.linalg.svd(X, full_matrices=False)
        components, score = svd_to_components_and_score(U, S, V_transpose, k)
        assert helpful_eq(components[0, 0], 0, failing_is_good=True)
        assert helpful_eq(components[0, 0], components[0, 1])
        assert helpful_eq(score[0, 0], -score[3, 0])
        assert helpful_eq(score[1, 0], -score[2, 0])
        np.random.seed(39)
        X = sk_scale(np.dot(np.random.rand(2, 2), np.random.randn(2, 100)).T)
        sk_pca = PCA(n_components=k)
        sk_score = sk_pca.fit_transform(X)
        sk_components = sk_pca.components_
        U, S, V_transpose = np.linalg.svd(X, full_matrices=False)
        components, score = svd_to_components_and_score(U, S, V_transpose, k)
        assert helpful_eq(sk_components, components)
        assert helpful_eq(sk_score, score)


def test_score_new_data():
    for k in [1, 2, 3]:

        X = sk_scale(np.array([[0.0, 0.4, -1, 1],
                               [1.0, 1.6, -2, 2],
                               [2.0, 3.4, -3, 3.5],
                               [3.0, 4.6, -4, 5]]))
        new_X = sk_scale(np.array([[0.5, 0.99, 1, -2],
                                   [2.5, 3.99, -42, -9]]))
        sk_pca = PCA(n_components=k)
        score = sk_pca.fit_transform(X)
        components = sk_pca.components_
        new_score = score_new_data(new_X, [1.42, 2.42, 4, 4], [1.11, 1.66, 2, 3], components)
        new_X_reconstr = new_score @ components
        if k == 1:
            assert helpful_eq(new_X_reconstr,
                              np.array([[-0.93533774, -0.93443902,  0.93533774, -0.93414063],
                                        [-0.09948932, -0.09939373,  0.09948932, -0.09936199]]))
        elif k == 2:
            assert helpful_eq(new_X_reconstr,
                              np.array([[-0.57078589, -1.07598312,  0.57078589, -1.52258937],
                                        [0.98203221, -0.51931478, -0.98203221, -1.84512183]]))
        elif k == 3:
            assert helpful_eq(new_X_reconstr,
                              np.array([[-0.34009009, -2.06024096,  0.34009009, -1.],
                                        [1.06081081, -0.85542169, -1.06081081, -1.66666667]]))


def test_mean_impute():
    X_train = np.array([[1, 2, np.nan],
                        [3, 4, 5]])
    X_test = np.array([[np.nan, 0, 3]])
    X_train, X_test = mean_impute(X_train, X_test)
    assert helpful_eq(X_train, [[1., 2., 5.], [3., 4., 5.]])
    assert helpful_eq(X_test, [[2., 0., 3.]])
    big_X, big_y, non_num_idx = create_big_X_y()
    X_train, X_test, _, _ = NFolds(big_X, big_y, seed=5).get_fold(0)
    X_train, X_test = mean_impute(X_train, X_test)
    assert helpful_eq(np.mean(X_train, axis=1)[0], 62.12441680772882)
    assert helpful_eq(np.mean(X_test, axis=1)[0], 53.21256512320771)
    assert helpful_eq(np.mean(X_train), 57.90520707104668)
    assert helpful_eq(np.mean(X_test), 58.20296278479702)


def test_only_num():
    X = np.array([[-1, 0, 1, 2],
                  [-3, 1, 4, 5]])
    assert helpful_eq(only_num(X, [1]), np.array([[-1, 1, 2],
                                                  [-3, 4, 5]]))
    big_X, big_y, non_num_idx = create_big_X_y()
    first_num_idx = min([i for i in range(big_X.shape[1]) if i not in non_num_idx])
    assert helpful_eq(only_num(big_X, non_num_idx)[0:2, 0], big_X[0:2, first_num_idx])


def test_replace_num_with_score():
    X = np.array([[1, 2.1, 4.3],
                  [0, 4.2, 5.3]])
    X = replace_num_with_score(X, [0], [[0,  1], [3,  2]])
    assert helpful_eq(X, [[1, 0, 1], [0, 3, 2]])
    X = np.array([[1, 2.1, 4.3]])
    X = replace_num_with_score(X, [0, 1], [[50]])
    assert helpful_eq(X, [[1, 2.1, 50]])


def test_repeated_cross_validation():
    X, y = load_breast_cancer(return_X_y=True)
    test_mean_impute()
    nn_idx = []
    accuracies = repeated_cross_validation(X, y, 2, 2, nn_idx, pca_k=False)
    assert helpful_eq(accuracies, [[0.91549296, 0.89824561],
                                   [0.9084507,  0.91578947]])
    accuracies_pca = repeated_cross_validation(X, y, 2, 2, nn_idx, pca_k=2)
    assert helpful_eq(accuracies_pca, [[0.91549296, 0.92631579],
                                       [0.92605634, 0.93684211]])
    X, y = load_breast_cancer(return_X_y=True)
    test_mean_impute()
    nn_idx = [0, 1, 2]
    accuracies = repeated_cross_validation(X, y, 4, 3, nn_idx, pca_k=False)
    assert helpful_eq(np.array(accuracies).shape, (4, 3))
    assert helpful_eq(accuracies, [[0.92592593, 0.89473684, 0.91052632],
                                   [0.8994709,  0.89473684, 0.92105263],
                                   [0.91005291, 0.89473684, 0.92631579],
                                   [0.92592593, 0.91052632, 0.91052632]])
    accuracies_pca = repeated_cross_validation(X, y, 4, 3, nn_idx, pca_k=2)
    assert helpful_eq(accuracies_pca, [[0.92063492, 0.88947368, 0.91578947],
                                       [0.88888889, 0.9,        0.9],
                                       [0.8994709,  0.92105263, 0.9],
                                       [0.93121693, 0.87894737, 0.91052632]])


def test_approx_X_from_X_with_score():
    X_train = np.array([[0.0, 0.4],
                        [1.0, 1.6],
                        [2.0, 3.4],
                        [3.0, 4.6]])
    X_test = np.array([[4, 5.6]])
    non_num_idx = []
    _, X_test_w_score, pca_dict = preprocess(X_train, X_test, non_num_idx, pca_k=1)
    assert helpful_eq(X_test_w_score, [[2.93797197]])
    # ^ tests only if mistake is in previous functions
    X_test_reconstr = approx_X_from_X_with_score(X_test_w_score, pca_dict, non_num_idx)
    assert helpful_eq(X_test_reconstr, [[3.82267078, 5.85623919]])
    assert helpful_eq(mean_squared_error(X_test, X_test_reconstr), 0.04855208630760682)

    X, y, non_num_idx = create_big_X_y()
    X_train, X_test, y_train, y_test = NFolds(X, y, seed=5).get_fold(0)
    X_train, X_test = mean_impute(X_train, X_test)
    _, X_test_w_score, pca_dict = preprocess(X_train, X_test, non_num_idx, pca_k=2)
    X_test_reconstr = approx_X_from_X_with_score(X_test_w_score, pca_dict, non_num_idx)
    assert helpful_eq(X_test_reconstr[0:5, -1],
                      [0.07272649, 0.08546382, 0.07856379, 0.07790174, 0.10962264])
    assert helpful_eq(mean_squared_error(X_test, X_test_reconstr), 2760.755214107711)
