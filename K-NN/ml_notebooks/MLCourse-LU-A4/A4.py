import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as sk_scale
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.utils import shuffle as sk_shuffle


def helpful_eq(a, b, failing_is_good=False):
    """Basically `==` after rounding with prints. 
    `a` and `b` can be numbers or (>1D) (NumPy) arrays."""
    def print_bad_news():
        print(a)
        print("helpful_eq(...) fail: ^ does not equal:")
        print(b)

    try:
        if hasattr(a, "__len__"):
            r = np.allclose(a, b, atol=1e-3)
        else:
            r = round(a, 3) == round(b, 3)
        if failing_is_good:
            r = not r
        if not r:
            print_bad_news()
        return r
    except Exception as e:
        print_bad_news()
        print("And/or we encountered exception message")
        print(e)
        return False


def mean_squared_error(true, pred):
    """`true` and `pred` should be (numpy.nd)arrays with similar shapes.
    Returns mean (over instances/rows) of sum of squared feature difference (in columns)."""
    return np.mean(((true - pred) ** 2).mean(axis=1))


class NFolds:
    def __init__(self, X, y, n_folds=5, seed=42):
        """ Initialize the KFolds instance

        :param X: numpy.ndarray of feature columns 
        :param y: numpy.ndarray of labels
        :param n_folds: number of folds desired
        :param seed: random seed, if you want reproducible results (optional)

        After initialization, self.folds will store n_folds folds.
        Each fold is a pair of arrays with training indices and test indices.
        The folds are as evenly distributed in size as possible.
        All the test segments are pairwise disjoint.
        """
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.folds = []
        indices = np.arange(X.shape[0])
        if seed is not None:
            np.random.seed(seed=seed)
        np.random.shuffle(indices)
        fold_size = X.shape[0] / n_folds
        for fold_num in range(n_folds):
            test = indices[int(fold_num * fold_size): int((fold_num + 1) * fold_size)]
            train = np.concatenate([indices[: int(fold_num * fold_size)],
                                    indices[int((fold_num + 1) * fold_size):]])
            self.folds.append((train, test))

    def get_fold(self, fold_num):
        """ Get the training and test data of the fold_num-th fold

        :param fold_num: Which fold's division of the data to use
        :return: Training and test features/labels
        """
        train, test = self.folds[fold_num]
        X_train = self.X[train]
        X_test = self.X[test]
        y_train = self.y[train]
        y_test = self.y[test]
        return X_train, X_test, y_train, y_test


def center_and_scale(X):
    """`X` is a numpy.ndarray. Rows represent instances. Columns represent feature values.
    First output is a modified copy of `X` where mean of individual features is 0 and 
    standard deviation is 1. The second and third output are original values of said mean and 
    standard deviation respectively."""
    raise NotImplementedError()


def uncenter_and_unscale(X_norm, orig_center, orig_std):
    """`X_norm` is a numpy.ndarray. Rows represent instances. Columns represent feature values,
    now with mean 0 and standard deviation 1. Output is a modified copy of `X_norm` with values of
    `orig_center` as column means and values of `orig_std` as column standard deviations. (This
    function thus reverses `center_and`scale`.)"""
    raise NotImplementedError()


def svd_to_components_and_score(U, Sigma_elements, V_transpose, k):
    """ Input is output of `numpy.linalg.svd(..., full_matrices=False)` and `k` indicating # of 
    components. First output is `k`-by-d component numpy.ndarray; second output is n-by-`k` score 
    numpy.ndarray.
    """
    raise NotImplementedError()


def score_new_data(new_X, orig_center, orig_std, components):
    """`new_X` is a numpy.ndarray. Rows represent instances. Columns represent feature values.
    First, create a modified copy of `new_X` as if normalizing it using `orig_center` and 
    `orig_std`. These params refer to feature means and standard deviations respectively. 
    Output is a representation of this normalized `new_X` in terms of `components`, i.e., a score 
    matrix."""
    raise NotImplementedError()


def pca(X, k):
    """All components are explained above. If you did not complete `center_and_scale` or 
    `svd_to_components_and_score`, you can fit in the scikit-learn alternatives exemplified 
    (in the tests) above."""
    X_norm, orig_center, orig_std = center_and_scale(X)
    U, Sigma_elements, V_transpose = np.linalg.svd(X_norm, full_matrices=False)
    S_squared_sum = sum(S_i ** 2 for S_i in Sigma_elements)
    R_squared = [S_i ** 2 / S_squared_sum for S_i in Sigma_elements]
    components, score = svd_to_components_and_score(U, Sigma_elements, V_transpose, k)
    return {
        'orig_center': orig_center,
        'orig_std': orig_std,
        'components': components,
        'score': score,
        'R_squared': np.array(R_squared)
    }


def create_big_X_y():
    """It's not necessary to understand completely what is happening, but notice that 
    the columns whose indices are in `non_num_idx` represent categorical features."""
    X, y = load_breast_cancer(return_X_y=True)

    # Add some instances
    X = np.repeat(X, 10, axis=0)
    y = np.repeat(y, 10)

    X, y = sk_shuffle(X, y, random_state=11)

    # Add two categorical features (that have a 60% prob of being right)
    np.random.seed(42)
    X = np.hstack((np.random.binomial(1, 0.4 + 0.2 * y).reshape(-1, 1),
                   np.random.binomial(1, 0.4 + 0.2 * y).reshape(-1, 1),
                   X))
    non_num_idx = [0, 1]

    # Add noise to numerical features
    num = [i for i in range(X.shape[1]) if i not in non_num_idx]
    X[:, num] += np.random.normal(0, 0.25 *
                                  X[:, num].std(axis=0), X[:, num].shape)

    # Replace some % of numerical values with np.nan
    X_num_flat = X[:, num].flatten()
    n_to_remove = int(len(X_num_flat) * .1)
    to_remove = np.random.permutation(range(len(X_num_flat)))[:n_to_remove]
    X_num_flat[to_remove] = np.nan
    X[:, num] = X_num_flat.reshape(X[:, num].shape)

    return X, y, non_num_idx


def mean_impute(X_train, X_test):
    """We assume numpy.ndarrays `X_train` and `X_test` only have numpy.nan as missing values in 
    their numerical features (columns). The output should be modified copies of `X_train` and 
    `X_test` where the missing values are replaced by the mean of the column of `X_train` in which 
    the value was missing. (The test might enlighten you.)"""
    raise NotImplementedError()


def only_num(X, non_num_idx):
    """The columns of numpy.ndarray `X` whose indices are stored in `non_num_idx` are not 
    numerical. Return a modified copy of `X` that does _not_ contain these columns."""
    raise NotImplementedError()


def replace_num_with_score(X, non_num_idx, score):
    """The columns of numpy.ndarray `X` whose indices are stored in `non_num_idx` are not 
    numerical. Return a modified copy of `X` that has unchanged non-numerical features but in 
    which the numerical features are replaced by the elements of the `score` numpy.ndarray."""
    # Assume only first features can be non-numerical to make our life easier
    assert all([i == non_num_idx[i] for i in range(len(non_num_idx))])

    raise NotImplementedError()


def preprocess(X_train, X_test, non_num_idx, pca_k=False):
    """Returns modified copies of `X_train` and `X_test` in which at least the missing values are
    imputed. If `pca_k` is not False but a positive integer, perform PCA as above, and replace
    numerical features (column indices not in `non_num_idx` with elements of the score matrix.
    The outcome of the `pca` function is returned as third value, or `None` if no PCA is done."""
    X_train, X_test = mean_impute(X_train, X_test)
    if pca_k > 0:
        pca_dict = pca(only_num(X_train, non_num_idx), pca_k)
        X_train_with_score = replace_num_with_score(X_train, non_num_idx, pca_dict['score'])
        X_test_s = score_new_data(only_num(X_test, non_num_idx), pca_dict['orig_center'],
                                  pca_dict['orig_std'], pca_dict['components'])
        X_test_with_score = replace_num_with_score(X_test, non_num_idx, X_test_s)
        return X_train_with_score, X_test_with_score, pca_dict
    else:
        return X_train, X_test, None


def repeated_cross_validation(X, y, n_repeats, n_folds, non_num_idx, pca_k=False):
    """For i = 0, ...`n_repeats`, create `X` and `y` folds with seed i. Then perform regular
    cross validation which includes `preprocess(X_train, X_test, non_num_idx, pca_k)` and training
    an `SVC(random_state=(i+1)*(j+1))` where `j` is the index of the test fold. Return an 
    `n_repeats`-by-`n_folds` numpy.ndarray of mean accuracies per test fold."""
    raise NotImplementedError()


def approx_X_from_X_with_score(X_with_score, pca_dict, non_num_idx):
    """Obtain the score matrix from the `non_num_idx` columns of `X_with_score`. Then reconstruct
    the feature values they originally represented using the PCA outcome stored in `pca_dict`. 
    Return these values in the same format as the original matrix, without forgetting the 
    `non_num_idx` columns."""
    raise NotImplementedError()
