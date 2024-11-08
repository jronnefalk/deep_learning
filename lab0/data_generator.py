import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self):
        self.x = []  # Initialize x as an empty list
        self.y = []  # Initialize y as an empty list

    def generate(self, K, N, sigma):
        # Arrays to hold the data points and labels
        data_points = []
        labels = []

        # Generate K clusters
        for i in range(K):
            # Generate a random mean for each cluster in the range [0, 1) for both dimensions
            mean = np.random.uniform(0, 1, 2)

            # Generate N points for this cluster, with the given mean and standard deviation
            cluster_points = np.random.normal(loc=mean, scale=sigma, size=(N, 2))

            # Append the points to the data array and the labels to the labels array
            data_points.append(cluster_points)
            labels.append([i] * N)  # Label all points in this cluster as i

        # Concatenate all clusters into a single dataset and store them in instance variables
        self.x = np.vstack(data_points)
        self.y = np.hstack(labels)

    def plot_data(self):
        # Determine the number of clusters K
        K = np.max(self.y) + 1

        plt.figure(figsize=(6, 6))
        for i in range(K):
            # Select points belonging to cluster i
            cluster_points = self.x[self.y == i]

            # Scatter plot for each cluster
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Class {i}")

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"2D Points for {K} Clusters")
        plt.legend()
        plt.show()

    def rotate(self, ang):
        # Define the rotation matrix
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

        # Apply the rotation to each point in self.x
        for i in range(len(self.x)):
            self.x[i, :] = R @ self.x[i, :]

    def export_data(self, filename):
        # Save x and y to a .npz file
        np.savez(filename, x=self.x, y=self.y)

    def import_data(self, filename):
        # Load x and y from a .npz file
        data = np.load(filename)
        self.x = data["x"]
        self.y = data["y"]
