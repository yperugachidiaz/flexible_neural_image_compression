import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set to "2" or "3" to suppress TensorFlow warnings

from PIL import Image
import io
import os
from glob import glob
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import tensorflow as tf
import tensorflow_datasets as tfds


def parse_example(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        # Define the feature description for parsing the TFRecord
        {
            "channels": tf.io.FixedLenFeature([], dtype=tf.string),
            "height": tf.io.FixedLenFeature([], dtype=tf.string),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "width": tf.io.FixedLenFeature([], dtype=tf.string)
        }
    )


# PyTorch DataLoader to create PyTorch dataset from .tfrecord files
class TFRecordPytorch(Dataset):
    def __init__(self, path_to_tfrecords, image_size=256, cycle_length=100, shuffle=True):
        self.path_to_tfrecords = path_to_tfrecords
        self.image_size = image_size
        self.cycle_length = cycle_length

        # Make list files
        ds_filenames = tf.data.Dataset.list_files(f"{self.path_to_tfrecords}/*.tfrecord", shuffle=shuffle)

        # Get the first file name
        self.ds = ds_filenames.interleave(tf.data.TFRecordDataset, cycle_length=self.cycle_length)
        self.ds = self.ds.map(parse_example)

        # Count number of examples in files
        self.num_examples = 1251470

        # Create iterator to loop over the dataset
        self.ds_iterator = iter(self.ds)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        while True:
            # Get the next example from the record iterator
            bytes_record = next(self.ds_iterator)

            # Decode the image bytes as a JPEG-encoded image
            numpy_bytes_array = bytes_record["image"].numpy()

            # Create numpy array from bytes_list by decoding
            # Add .convert('RGB') to convert (H, W) to (3, H, W)
            image = Image.open(io.BytesIO(numpy_bytes_array)).convert('RGB')

            # Skip images < than crop size
            if min(image.size) >= self.image_size:
                break

        transform = transforms.Compose([
            # RandomCrop to be sure no images < crop size
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        return transform(image)


def get_imagenet_from_tfrecords(path_train_data, image_size=256, cycle_length=100, shuffle=True):
    train_dataset = TFRecordPytorch(path_train_data, image_size, cycle_length, shuffle)
    return train_dataset


class CLICDataset(Dataset):
    def __init__(self, path_to_clic, image_size=256, shuffle=True):
        self.path_to_clic = path_to_clic
        self.image_size = image_size
        self.shuffle = shuffle

        # Load data
        dataset = tfds.load("clic", split="train", data_dir=self.path_to_clic, shuffle_files=self.shuffle)

        # Convert to numpy arrays dataset
        self.numpy_dataset = tfds.as_numpy(dataset)

        # Create iterator to loop over the dataset
        self.clic_iter = iter(self.numpy_dataset)

    def __len__(self):
        # Fixed to speed up process
        return 1633

    def __getitem__(self, idx):
        # Iterate over the iterable dataset
        example = next(self.clic_iter)

        numpy_array = example["image"]
        # Create numpy array from bytes_list by decoding
        image = Image.fromarray(numpy_array)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        return transform(image)


def get_clic_from_tfrecords(path_train_data, image_size=256, shuffle=True):
    train_dataset = CLICDataset(path_train_data, image_size, shuffle)
    return train_dataset


# Combined dataset that interleaves samples from both datasets
class CombinedDataset(Dataset):
    def __init__(self, imagenet_dataset, clic_dataset):
        self.imagenet_dataset = imagenet_dataset
        self.clic_dataset = clic_dataset

    def __len__(self):
        return len(self.imagenet_dataset) + len(self.clic_dataset)

    def __getitem__(self, idx):
        if idx < len(self.imagenet_dataset):
            # Within the range of TFRecords dataset
            return self.imagenet_dataset[idx]
        else:
            # Beyond the range of TFRecords dataset, sample from CLIC dataset
            return self.clic_dataset[idx - len(self.imagenet_dataset)]


def get_combined_dataset(path_imagenet, path_clic, image_size, cycle_length=100, shuffle=True):
    # Load datasets
    train_clic = get_clic_from_tfrecords(path_clic, image_size, shuffle=shuffle)
    train_imagenet = get_imagenet_from_tfrecords(path_imagenet, image_size, cycle_length, shuffle=shuffle)

    # Combine datasets
    combined_dataset = CombinedDataset(imagenet_dataset=train_imagenet, clic_dataset=train_clic)
    return combined_dataset


class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} does not exist")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


class TestPathologyDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} does not exist")
        # Include only .tif files (.png are binary masks)
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.tif")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


class TestTecnickDataset(Dataset):
    # Downloaded from https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip
    # The used sub-folder is "./RGB/RGB_OR_1200x1200"
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} does not exist")

        # Include the .png files as they contain the images
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.png")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)
