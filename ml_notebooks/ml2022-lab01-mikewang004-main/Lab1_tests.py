import numpy as np
import pandas as pd
import pytest

import Lab1

# DO NOT CHANGE THIS FILE!!!


def test_plus_one():
    assert Lab1.plus_one(1) == 2
    assert Lab1.plus_one(10) == 11
    with pytest.raises(TypeError):
        Lab1.plus_one('two')


def test_hello():
    world = Lab1.hello('World')
    assert world == 'Hello World!', f'"{world}" != "Hello World!"'
    you = Lab1.hello('you')  # sneaky, the first letter needs to be capitalized in the output
    assert you == 'Hello You!', f'"{you}" != "Hello You!"'


def test_roll_array():
    a = np.arange(10)

    assert np.array_equal(
        Lab1.roll_array(a, 2, 'right'),
        np.asarray([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    )

    assert np.array_equal(
        Lab1.roll_array(a, 4, 'left'),
        np.asarray([4, 5, 6, 7, 8, 9, 0, 1, 2, 3])
    )


def test_surface_class():
    df = pd.DataFrame({
        'height': [1, 2, 3, 4, 5],
        'width': [5, 4, 8, 10, 1],
        'class_labels': ['a', 'b', 'c', 'a', 'a']
    })

    labels = np.asarray(Lab1.surface_class(df, 8))
    target = np.asarray(['b', 'c', 'a'])
    assert np.array_equal(labels, target), f"{labels} != {target}"

    labels = np.asarray(Lab1.surface_class(df, 9))
    target = np.asarray(['c', 'a'])
    assert np.array_equal(labels, target), f"{labels} != {target}"

    labels = np.asarray(Lab1.surface_class(df, 24))
    target = np.asarray(['c', 'a'])
    assert np.array_equal(labels, target), f"{labels} != {target}"

    labels = np.asarray(Lab1.surface_class(df, 25))
    target = np.asarray(['a'])
    assert np.array_equal(labels, target), f"{labels} != {target}"


def test_NNClassifier():
    # Make some training data
    train = Lab1.generate_data()
    x_train = train[['height', 'width']]
    y_train = train['label']

    # Make some new data to test with
    test = Lab1.generate_data(random_state=43)  # we want different test data
    x_test = test[['height', 'width']]
    y_test = test['label']

    classifier = Lab1.OneNearestNeighborClassifier()  # Initialize
    classifier.fit(x_train, y_train)  # Learn from the training data

    prediction = classifier.predict(x_test)  # Make a prediction about the test data

    accuracy = np.sum(y_test == prediction) / len(y_test)
    target_accuracy = 0.75
    assert accuracy >= target_accuracy, f'Accuracy {accuracy} != {target_accuracy}'


def test_NNC_imbalanced():
    # Make some training data
    train = Lab1.generate_data()
    x_train = train[['height', 'width']]
    y_train = train['label']

    # Make some new data to test with
    test = Lab1.generate_data(random_state=43, proportion=0.3)  # switch to class b as the most common
    x_test = test[['height', 'width']]
    y_test = test['label']

    clf = Lab1.OneNearestNeighborClassifier()  # Initialize
    clf.fit(x_train, y_train)  # Learn from the training data

    prediction = clf.predict(x_test)  # Make a prediction about the test data
    accuracy = np.sum(y_test == prediction) / len(y_test)
    target_accuracy = 0.65
    assert accuracy >= target_accuracy, f'Accuracy {accuracy} != {target_accuracy}'