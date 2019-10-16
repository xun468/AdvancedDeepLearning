from keras.datasets import cifar10
import numpy as np

from utils import display_image


class Generator:
    def __init__(self):
        self.means = []
        self.covs = []
        self.orig_shape = None

    def calc_class_stats(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.orig_shape = x_train[0].shape

        for k in range(10):
            bol = (y_train == k).reshape(-1)
            sub_x = x_train[bol]
            mean = np.mean(sub_x, axis=0).flatten()
            sub_x = sub_x.reshape(sub_x.shape[0], -1)
            cov = (sub_x - mean).T @ (sub_x - mean) / len(mean)

            self.means.append(mean)
            self.covs.append(cov)

    def generate_seed(self, class_id):
        sample = np.rint(np.random.multivariate_normal(self.means[class_id], self.covs[class_id])).astype(int).reshape(
            self.orig_shape)
        sample = np.clip(sample, 0, 255)
        return sample

    def visualize_example_seeds(self, count=3):
        for k in range(count):
            sample = self.generate_seed(k)
            display_image(sample, "Example seed from class {}".format(k))
