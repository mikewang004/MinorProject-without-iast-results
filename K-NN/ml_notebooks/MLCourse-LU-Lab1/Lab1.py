import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CHANGE ONLY THIS FILE!!!

def plus_one(number):
    """ Adds 1, if 'number' is a number. """
    return 1 + number


def hello():
    """ Should return 'Hello Text!' """
    raise NotImplementedError('Your code here')


def roll_array(array, steps, direction='left'):
    """ Roll the contents of a 1D array, "steps" steps in the given direction, wrapping around. """
    raise NotImplementedError('Your code here')
    return rolled_array


def surface_class(frame, min_surface):
    """ Return the class labels of all rows with a surface >= least min_surface """
    raise NotImplementedError('Your code here')
    return labels


class OneNearestNeighborClassifier:
    def __init__(self):
        self.y_train = None
        self.x_train = None

    def fit(self, x_train, y_train):
        """ To fit, we just store the entire training data. """
        self.x_train = x_train
        self.y_train = y_train

    def distance(self, a, b):
        """ Return the Euclidean distance between point a and b """
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, x_test):
        """ Predict the label of test instances

        For each test instance, find the nearest point in the training set.
        Then return a list of the labels of the nearest points that you found.
        """
        prediction = []

        raise NotImplementedError('Your code here')

        if isinstance(x_test, pd.DataFrame):
            return pd.Series(prediction)
        return np.asarray(prediction)  # as a numpy array


def generate_data(
        amount=100, proportion=0.7,  # By default we want 70% class a, 30% class b
        signal_a_height=5, signal_a_width=10, a_noise=2,
        signal_b_height=4, signal_b_width=6, b_noise=3,
        random_state=42  # by default, always generate the same data
):
    """ Generate some synthetic data """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate how many instances of each class to generate
    a_amount = int(amount * proportion)
    b_amount = amount - a_amount

    # Generate instances of class a
    # The height and width are the sum of signal + noise
    a_height = signal_a_height + (np.random.randn(a_amount) * a_noise)
    a_width = signal_a_width + (np.random.randn(a_amount) * a_noise)
    a_labels = np.full(a_amount, 'a')

    # Do the same for b
    b_height = signal_b_height + (np.random.randn(b_amount) * b_noise)
    b_width = signal_b_width + (np.random.randn(b_amount) * b_noise)
    b_labels = np.full(b_amount, 'b')

    # Wrap it up in a nice dataframe
    df = pd.DataFrame({
        'height': np.concatenate([a_height, b_height]),
        'width': np.concatenate([a_width, b_width]),
        'label': np.concatenate([a_labels, b_labels])
    })
    return df


def visualize(x_test, y_test, prediction, classifier_name=None):
    """ Visualize classifier predictions vs. ground truth """
    # Initialize some things
    plt.figure(figsize=(10, 10))
    color_map = {'b': 'red', 'a': 'blue'}
    marker_map = {'a': '+', 'b': 'o'}

    # place the scatter points one by one
    for i in range(len(prediction)):
        y_coord = x_test.iloc[i, 0]  # height
        x_coord = x_test.iloc[i, 1]  # width
        true_label = y_test.iloc[i]
        predicted_label = prediction.iloc[i]
        plt.scatter(
            x_coord, y_coord,
            color=color_map[predicted_label], marker=marker_map[true_label],
            label=f'predicted {predicted_label} / true {true_label}'
        )

    # this is a trick to make sure we don't get duplicate handles in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f'Predictions+Ground Truth using the {classifier_name} classifier')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()