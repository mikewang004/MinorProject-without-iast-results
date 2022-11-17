import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

if __name__ == '__main__':

    """Task 1
    Implement code that does the following:
    - Apply KMeans and KMedoids using k=5
    - Make three plots side by side, showing the clusters identified by the models and the ground truth
    - Plots should have titles, axis labels, and a legend.
    - The plots should also show the centroids of the KMeans and KMedoids clusters.
      Use a different marker style to make them clearly identifiably.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/vehicles.csv'

    # TODO: student code here

    plt.tight_layout()
    plt.savefig('Figure1.pdf')  # save as PDF to get the nicest resolution in your report.
    plt.show()

    """ Task 2
    Apply KMeans and KMedoids to the following dataset. The choice of K is up to you.
    - Make plots of the best results you got with KMeans and KMedoids.
    - In the title of the plots, indicate the K used, and the homogeneity and completeness score achieved.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-2.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.iloc[:, :-1].values  # all except the last column
    y = df.iloc[:, -1].values  # the last column
    feature_names = df.columns[:-1]
    k = 4

    # TODO: student code here

    plt.savefig('Figure2.pdf')
    plt.show()

    """ Task 3

    Adapt the code used in the example to instead make a comparison between KMeans and KMedoids.
    - Set K at 4
    - Make a plot for both models
    """
    url = f'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-3.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.values  # convert from pandas to numpy
    n_clusters = 4

    # TODO: student code here

    """ FINISH """
    plt.savefig('Figure3.pdf')
    plt.show()  # show all the plots, in the order they were generated

    """Task 4

    Write code to generate elbow plots for the datasets given here.
    Use them to figure out the likely K for each of them.
    Put the plots you used to make your decision in your report.
    """
    for dataset in ['4a', '4b']:
        url = f'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-{dataset}.csv'
        df = pd.read_csv(filepath_or_buffer=url, header=0)
        X = df.values

        # TODO: student code here

        plt.tight_layout()
        plt.savefig(f'Figure 4-{dataset}.pdf')
        plt.show()

    """ Task 5

    Write code that generates a dataset with k >= 3 and 2 feature dimensions.
    - It should be easy for a human to cluster with the naked eye.
    - It should NOT be easy for KMedoids to cluster, even when using the correct value of K.
    - Plot the ground truth of your dataset, so that we can see that a human indeed clusters it easily.
    - Plot the clustering found by KMedoids to show that it doesn't do it well.
    """

    # TODO: student code here

    plt.tight_layout()
    plt.savefig('Figure5.pdf')
    plt.show()

