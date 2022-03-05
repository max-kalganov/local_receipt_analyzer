"""Downloads SROIE dataset"""
from typing import Generator

import tensorflow as tf
from doctr.datasets import SROIE
import cv2
import os
import gin
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ""


@gin.configurable
class DatasetGenerator(Generator):
    def __init__(self, train: bool = True, datagen_batch_size: int = 1, *args, **kwargs):
        self._image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            *args, *kwargs
            # TODO: add params using gin
        )
        assert self._datagen_batch_size > 0
        self._datagen_batch_size = datagen_batch_size
        self._train = bool(train)

    def __iter__(self):
        batch_x = []
        batch_y = []
        for img, target in SROIE(train=self._train, download=True):
            batch_x.append(img)
            batch_y.append((target['label'], target['boxes']))
            if len(batch_x) == self._datagen_batch_size:
                batch_x_np = np.concatenate(batch_x, axis=0)
                for img_proc, target_proc in self._image_datagen.flow(batch_x_np.reshape((-1,
                                                                                          batch_x_np.shape[1],
                                                                                          batch_x_np.shape[2],
                                                                                          1)),
                                                                      batch_y,
                                                                      batch_size=self._datagen_batch_size):
                    yield img_proc, target_proc
                batch_x, batch_y = [], []
        if len(batch_x) != 0:
            batch_x_np = np.concatenate(batch_x, axis=0)
            for img_proc, target_proc in self._image_datagen.flow(batch_x_np.reshape((-1,
                                                                                      batch_x_np.shape[1],
                                                                                      batch_x_np.shape[2],
                                                                                      1)),
                                                                  batch_y,
                                                                  batch_size=self._datagen_batch_size):
                yield img_proc, target_proc


def get_train_dataset() -> DatasetGenerator:
    return DatasetGenerator(train=True)


def get_test_dataset() -> DatasetGenerator:
    return DatasetGenerator(train=False)


def show_dataset():
    train_set = SROIE(train=True, download=True)
    for i, train_example in enumerate(train_set):
        if i % 300 != 0:
            continue
        img, target = train_example
        img_numpy = img.numpy()
        for example in target['boxes']:
            unnormalized_example = [int(example[0] * img.shape[1]), int(example[1] * img.shape[0]),
                                    int(example[2] * img.shape[1]), int(example[3] * img.shape[0])]
            cv2.rectangle(img=img_numpy,
                          pt1=(unnormalized_example[0], unnormalized_example[1]),
                          pt2=(unnormalized_example[2], unnormalized_example[3]),
                          color=(0, 0, 255), thickness=2)

        cv2.namedWindow(f"examples_{i}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"examples_{i}", 500, 700)
        cv2.imshow(winname=f"examples_{i}", mat=img_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
