"""normalized fracatlas dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image

from fracatlas.utils import find_bbox


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for fracatlas dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(
                        names=["Fractured", "Non_fractured"]
                    ),
                    "bbox": [tfds.features.BBox()],
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://doi.org/10.1038/s41597-023-02432-4",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(fracatlas): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://figshare.com/ndownloader/files/41725659"
        )

        # TODO(fracatlas): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train_imgs"),
            "test": self._generate_examples(path / "test_imgs"),
        }

    def _generate_examples(self, split_path):
        """Yields examples."""
        # TODO(fracatlas): Yields (key, example) tuples from the dataset

        for img_path in split_path.glob("../images/*/*.JPG"):
            im = Image.open(img_path)
            resized_image = tf.image.resize_with_pad(im, 225, 225)
            tf.keras.preprocessing.image.save_img(split_path, resized_image)

            bbox = find_bbox(
                img_path,
                split_path / ".." / "Annotations" / "PASCAL VOC",
                target_size=(256, 256),
            )

            yield img_path.name, {
                "image": img_path,
                "label": "Fractured"
                if img_path.parent.name == "Fractured"
                else "Non_fractured",
                "bbox": bbox,
            }
