import os

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from yaml import CLoader as Loader


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, transform, split):
        """Initializes a pytorch Dataset object
        :param data_dir: A filename (string), to identify the yaml file
          containing the dataset.
        :param transform: Transformation function to be applied to the input
          images (e.g. created with torchvision.transforms.Compose()).
        :param split: A list of strings, one for each dataset split to be
          loaded by the Dataset object.
        """

        self.data_dir = data_dir
        self.transform = transform
        self.imgs = []
        self.lbls = []

        with open(self.data_dir, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        image_folder = data_dir.replace("isic_classification.yml", "")

        for s in split:
            for i in d['split'][s]:
                self.imgs.append(
                    os.path.join(image_folder, d['images'][i]['location']))
                self.lbls.append(d['images'][i]['label'])

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, self.lbls[index]

    def __len__(self):
        return len(self.lbls)


def get_data_loaders_classification(data_dir, train_batch_size, test_batch_size, shuffle, num_workers, pin_memory):
    normalize = transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = os.path.join(data_dir)

    train_dataset = ClassificationDataset(data_dir, transform_train, ["training"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=pin_memory)

    valid_dataset = ClassificationDataset(data_dir, transform_test, ["validation"])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=pin_memory)

    test_dataset = ClassificationDataset(data_dir, transform_test, ["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle,
                                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    tr, v, te = get_data_loaders_classification("D:\\datasets\\isic_classification\\isic_classification.yml", 12, 12,
                                                True, 8, True)

    print(len(tr), len(v), len(te))

    from tqdm import tqdm

    for i, l in tqdm(te):
        pass
