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
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    k = 5
    modelmeans = KMeans(n_clusters=k)
    modelmedoids = KMedoids(n_clusters=k)
    X = df.iloc[:, :-1]  # all except the last column
    y = df.iloc[:, -1]  # the last column
    modelmeans.fit(X)
    modelmedoids.fit(X)
    x = df['weight']
    y = df['speed']
    c = modelmeans.labels_
    d = modelmedoids.labels_
    e = df['label'] - 1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), sharex=True, sharey=True)
    ax1.scatter(x, y, c=c)
    ax1.scatter(modelmeans.cluster_centers_[:, 0], modelmeans.cluster_centers_[:, 1], marker='x', color='red')
    ax2.scatter(x, y, c=d)
    ax2.scatter(modelmedoids.cluster_centers_[:, 0], modelmedoids.cluster_centers_[:, 1], marker='x', color='red')
    ax3.scatter(x, y, c=e)

    ax1.set_title('Clustering for KMeans')
    ax1.set_xlabel('weight')
    ax1.set_ylabel('speed')
    ax1.legend(*ax1.scatter(x=x, y=y, c=c).legend_elements())

    ax2.set_title('Clustering for KMedoids')
    ax2.set_xlabel('weight')
    ax2.set_ylabel('speed')
    ax2.legend(*ax2.scatter(x=x, y=y, c=d).legend_elements())

    ax3.set_title('Clustering, ground truth')
    ax3.set_xlabel('weight')
    ax3.set_ylabel('speed')
    ax3.legend(*ax3.scatter(x=x, y=y, c=e).legend_elements())

    plt.suptitle(f"Plots to compare KMeans and KMedoids clustering, x denote cluster centres",
                 fontsize=14)

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

    modelmeans = KMeans(n_clusters=k)
    modelmedoids = KMedoids(n_clusters=k)
    modelmeans.fit(X)
    modelmedoids.fit(X)
    # print(y)
    # print('centroids:\n', modelmeans.cluster_centers_, '\n')
    # print('labels:\n', modelmeans.labels_)
    # print('centroids:\n', modelmedoids.cluster_centers_, '\n')
    # print('labels:\n', modelmedoids.labels_)

    # plotting models

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), sharex=True, sharey=True)

    ax1.scatter(X[:, 0], X[:, 1], c=modelmeans.labels_)
    ax1.scatter(modelmeans.cluster_centers_[:, 0], modelmeans.cluster_centers_[:, 1], marker='x', color='red')
    ax1.set_title("Plot, model used is k-means")
    ax1.set_xlabel("X0")
    ax1.set_ylabel("X1")
    ax1.legend(*ax2.scatter(x=X[:, 0], y=X[:, 1], c=modelmeans.labels_).legend_elements(), loc="upper left")

    ax2.scatter(X[:, 0], X[:, 1], c=modelmedoids.labels_)
    ax2.set_title("Plot, model used is k-medoids")
    ax2.scatter(modelmedoids.cluster_centers_[:, 0], modelmedoids.cluster_centers_[:, 1], marker='x', color='red')
    ax2.set_xlabel("X0")
    ax2.set_ylabel("X1")
    ax2.legend(*ax2.scatter(x=X[:, 0], y=X[:, 1], c=modelmedoids.labels_).legend_elements(), loc="upper left")

    ax3.scatter(X[:, 0], X[:, 1], c=y, label=y)
    ax3.set_title("Plot of true values")
    ax3.set_xlabel("X0")
    ax3.set_ylabel("X1")
    ax3.legend(*ax1.scatter(x=X[:, 0], y=X[:, 1], c=y).legend_elements(), loc="upper left")

    plt.suptitle(f"Plots to compare KMeans and KMedoids clustering, x denote cluster centres",
                 fontsize=14)


    # get homogenity and completeness scores

    mean_homogeneity = homogeneity_score(y, modelmeans.labels_)
    medoids_homogeneity = homogeneity_score(y, modelmedoids.labels_)
    mean_completeness = completeness_score(y, modelmeans.labels_)
    medoids_completeness = completeness_score(y, modelmedoids.labels_)

    print("Values are left to right homogeinity respectively completeness scores,")
    print("top to bottom KMeans respectively KMedoids.")
    print(mean_homogeneity, mean_completeness)
    print(medoids_homogeneity, medoids_completeness)

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

    k = n_clusters

    # Train models

    modelmeans = KMeans(n_clusters=k, random_state=None)
    modelmeans_labels = modelmeans.fit_predict(X)
    modelmeans_centers = modelmeans.cluster_centers_
    modelmedoids = KMedoids(n_clusters=k, random_state=None)
    modelmedoids_labels = modelmedoids.fit_predict(X)
    modelmedoids_centers = modelmedoids.cluster_centers_


    # Compute the silhouette scores for each sample

    def model_comparison_plot(k, X, model_name, labels, centers):

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, labels)
        print(f"For n_clusters = {k}, the average silhouette_score is :{silhouette_avg}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        y_lower = 10  # starting position on the y-axis of the next cluster to be rendered
        sample_silhouette_values = silhouette_samples(X, labels)
        for i in range(k):  # Here we make the colored shape for each cluster

            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            # Figure out how much room on the y-axis to reserve for this cluster
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            y_range = np.arange(y_lower, y_upper)

            # Use matplotlib color maps to make each cluster a different color, based on the total number of clusters.
            # We use this to make sure the colors in the right plot will match those on the left.
            color = cm.nipy_spectral(float(i) / k)

            # Draw the cluster's overall silhouette by drawing one horizontal stripe for each datapoint in it
            ax1.fill_betweenx(y=y_range,  # y-coordinates of the stripes
                              x1=0,  # all stripes start touching the y-axis
                              x2=ith_cluster_silhouette_values,  # ... and they run as far as the silhouette values
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            # The 1st subplot is the silhouette plot
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")

            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylabel("Cluster label")
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            ax1.set_yticks([])  # Clear the yaxis labels / ticks

            ### RIGHT PLOT ###

            # 2nd Plot showing the actual clusters formed
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                f"Silhouette analysis for {model_name} clustering on sample data with n_clusters = {n_clusters}",
                fontsize=14, fontweight='bold')

            colors = cm.nipy_spectral(labels.astype(float) / k)  # make the colors match with the other plot
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

            # Labeling the clusters
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
            # Put numbers in those circles
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')



    #model_comparison_plot(k, X, "KMeans", modelmeans_labels, modelmeans_centers)
    model_comparison_plot(k, X, "KMedoids", modelmedoids_labels, modelmedoids_centers)

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

        k_max = 6
        k_values = np.arange(1, k_max + 1)
        url1 = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-4a.csv'
        url2 = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-4b.csv'
        df1 = pd.read_csv(filepath_or_buffer=url1, header=0)
        df2 = pd.read_csv(filepath_or_buffer=url2, header=0)
        X = df1
        Y = df2
        # Calculate values of inertia
        inertia = np.zeros([4, k_max])
        for k in range(1, k_max + 1):
            modelmeans = KMeans(n_clusters=k)  # fitting model
            modelmedoids = KMedoids(n_clusters=k)
            modelmeans.fit(X)
            modelmedoids.fit(X)
            inertia[0, k - 1], inertia[1, k - 1] = modelmeans.inertia_, modelmedoids.inertia_

            modelmeans2 = KMeans(n_clusters=k)
            modelmedoids2 = KMedoids(n_clusters=k)
            modelmeans2.fit(Y)
            modelmedoids2.fit(Y)
            inertia[2, k - 1], inertia[3, k - 1] = modelmeans2.inertia_, modelmedoids2.inertia_

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Plot inertia
        ax1.plot(k_values, inertia[0, :])
        ax1.set_title("Elbow method with Kmeans and inertia")
        ax1.set_xlabel("k")
        ax1.set_ylabel("inertia")

        ax2.plot(k_values, inertia[1, :])
        ax2.set_title("Elbow method with Kmedoids and inertia")
        ax2.set_xlabel("k")
        ax2.set_ylabel("inertia")

        plt.suptitle(f"Elbow method for dataset 4a", fontsize=14)
        plt.savefig(f'Figure 4.pdf')

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.plot(k_values, inertia[2, :])
        ax1.set_title("Elbow method with Kmeans and inertia")
        ax1.set_xlabel("k")
        ax1.set_ylabel("inertia")

        ax2.plot(k_values, inertia[3, :])
        ax2.set_title("Elbow method with Kmedoids and inertia")
        ax2.set_xlabel("k")
        ax2.set_ylabel("inertia")

        plt.suptitle(f"Elbow method for dataset 4b", fontsize=14)

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

    # Generate dataset

    k = 4

    X, y = make_blobs(n_samples=800,
                      n_features=2,
                      centers=k,
                      cluster_std=0.9,
                      center_box=(-3.25, 3.25),
                      shuffle=True,
                      random_state=11)  # For reproducibility, random_state = 11

    #Random_state is set now but if set to None this should reliably generate a dataset
    #with which k-medoids has issues with. 
    np.savetxt("A3.csv", X)
    modelmedoids = KMedoids(n_clusters=k)
    modelmedoids.fit(X)

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax1.set_title((f"Random dataset, ground truth, k = {k}"))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(*ax1.scatter(x=X[:, 0], y=X[:, 1], c=y).legend_elements(), loc="upper right")

    ax2.scatter(X[:, 0], X[:, 1], c=y)
    ax2.set_title((f"Random dataset, KMedoids fit, k = {k}"))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(*ax2.scatter(x=X[:, 0], y=X[:, 1], c=modelmedoids.labels_).legend_elements(), loc="upper right")

    plt.suptitle(f"Comparison of ground truth vs KMedoids for a 'bad' dataset", fontsize=14)

    plt.tight_layout()
    plt.savefig('Figure5.pdf')
    plt.show()

