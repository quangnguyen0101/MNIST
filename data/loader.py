# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, invalid-name
import struct
from array import array

# from os.path import join
# import random
import numpy as np


class MnistDataloader:  # MNIST Data Loader Class
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(size, rows, cols)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     input_path = "data/raw/"
#     train_img_path = join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte")
#     train_lbl_path = join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
#     test_img_path = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
#     test_lbl_path = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
#
#     def show_images(images, title_texts):
#         cols = 5
#         rows = int(len(images) / cols) + 1
#         plt.figure(figsize=(40, 30))
#         index = 1
#         for x in zip(images, title_texts):
#             image = x[0]
#             title_text = x[1]
#             plt.subplot(rows, cols, index)
#             plt.imshow(image, cmap="gray")
#             if title_text != "":
#                 plt.title(title_text, fontsize=15)
#             index += 1
#
#     loader = MnistDataloader(
#         train_img_path,
#         train_lbl_path,
#         test_img_path,
#         test_lbl_path,
#     )
#     (train_imgs, train_lbls), (test_imgs, test_lbls) = loader.load_data()
#
#     images_2_show = []
#     titles_2_show = []
#     for i in range(0, 10):
#         r = random.randint(1, 60000)
#         images_2_show.append(train_imgs[r])
#         titles_2_show.append(f"training image [{r}] = {train_lbls[r]}")
#
#     for i in range(0, 5):
#         r = random.randint(1, 10000)
#         images_2_show.append(test_imgs[r])
#         titles_2_show.append(f"test image [{r}] = {test_lbls[r]}")
#
#     show_images(images_2_show, titles_2_show)
#     plt.show()
