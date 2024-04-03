import os
import torch
import random
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import pytz
from datetime import datetime


def load_vae_data_from_disk(
        data_path='data/vae',
        batch_size=1,
        shuffle=False,
        image_key='image',
        drop_last=True,
        pin_memory=False
    ):
    # Load data
    total_data = []
    for name in os.listdir(data_path):
        file_path = os.path.join(data_path, name)
        if os.path.isfile(os.path.join(data_path, name)):
            data = torch.load(os.path.join(data_path, name))

            for key, value in data.items():
                images = value['images']

                for image in images:
                    if len(image.shape) == 4 and image.shape[0] == 1:
                        image = image.squeeze(0)

                    total_data.append(
                        {
                            image_key: image.contiguous(),
                            'path': file_path,
                            'dataset_name': name,
                            'object_name': key
                        }
                    )

    # Shuffle data
    dataloader = DataLoader(
        total_data,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

    return dataloader


def load_multi_dataset(
        data_structure,         
        saving_path=None
    ):
    """
    Load multiple dataset from a dict format
    :data_structure: Structure to load data
    {
        'First dataset name': {
            'dataset_path': dataset path,
            'splits': ['train', 'test'],
            'num_data': number of data per split
        },
        'Second dataset name': {
            'dataset_path': dataset path,
            'splits': ['...'],
            'num_data': number of data per split
        }
    }

    :saving_path: whether to save data or not
    :return: Tensor of data
    """ 

    datas = {}
    for key, value in data_structure.items():
        dataset_path = value['dataset_path']
        splits = value['splits']
        num_data = value['num_data']

        if 'valid_name' in value:
            valid_name = value['valid_name']
        else:
            valid_name = []

        if 'invalid_name' in value:
            invalid_name = value['invalid_name']
        else:
            invalid_name = []

        data = load_dataset(
            dataset_path, 
            splits,
            valid_name=valid_name,
            invalid_name=invalid_name,
            num_data=num_data
        )

        datas[key] = data

        # Save data
        if saving_path:
            torch.save(data, os.path.join(saving_path, f'{key}.pt'))

    return datas


def load_dataset(
        dataset_path: str, 
        splits: list=[],
        valid_name: list=[],
        invalid_name: list=[],
        num_data: int=0
    ):
    object_split_name = {}
    object_names = [
        name for name in 
        os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, name))
    ]
    data = {}

    for object_name in object_names:
        # Check valid split
        if splits:
            valid_split = [
                split for split in 
                os.listdir(os.path.join(dataset_path, object_name)) 
                if (split in splits) and os.path.isdir(os.path.join(dataset_path, object_name, split))
            ]
        else:
            valid_split = [
                split for split in 
                os.listdir(os.path.join(dataset_path, object_name)) 
                if os.path.isdir(os.path.join(dataset_path, object_name, split))
            ]

        # Load data
        for split in valid_split:
            images, paths = load_images(
                [os.path.join(dataset_path, object_name, split)],
                valid_name=valid_name,
                invalid_name=invalid_name,
                num_data=num_data
            )
            data[object_name] = {
                'images': images,
                'paths': paths
            }

    return data
        
 
def load_images(
        dir_list: list, 
        reshape_image: list=[512, 512], 
        extension_list: list=[], 
        valid_name: list=[],
        invalid_name: list=[],
        saving_path: str=None,
        num_data: int=0
    ):
    path_list = []
    images = []

    # Get image path
    for folder in dir_list:
        file_names = os.listdir(folder)


        # Get random number of image
        if num_data and len(file_names) >= num_data:
            selected_images = random.sample(file_names, num_data)
        else:
            selected_images = file_names
        

        for file_name in tqdm(selected_images):
            relative_path, file_ext = os.path.splitext(file_name)

            # Check valid name
            if valid_name:
                valid_check = False
                for name in valid_name:
                    if name in file_name:
                        valid_check = True
                        break 
                
            else:
                valid_check = True

            # Check invalid name
            if invalid_name:
                invalid_check = True
                for name in invalid_name:
                    if name in file_name:
                        invalid_check = False
                        break 
            else:
                invalid_check = True

            # Check file extension
            if valid_check and invalid_check:
                if extension_list:
                    if file_ext in extension_list:
                        path_list.append(os.path.join(folder, file_name))
                else:
                    path_list.append(os.path.join(folder, file_name))

    for path in path_list:

        # Check file path
        if os.path.isfile(path):
            # Load image
            image = Image.open(path)

            # Define transformation
            transform = transforms.Compose([
                transforms.Resize(reshape_image),
                transforms.ToTensor()            
            ])

            # Convert image to tensor
            tensor_image = transform(image)

            # Blend A to RGB
            if tensor_image.shape[0] == 4:
                tensor_image = tensor_image[:3, :] * tensor_image[-1:, :] + (1 - tensor_image[-1:, :])
            elif tensor_image.shape[0] < 3 and tensor_image.shape[0] > 4:
                raise ValueError(f"Number of channel invalid !!! Get {tensor_image.shape[0]} channel")

            images.append(tensor_image)

    # (Batch, Channel, Height, Width)
    if images:
        images = torch.stack(images, 0)

    return images, path_list

def plot_latent_space(vae=None, n=10, img_size=512, scale=0.5, figsize=5):
    # display a n*n 2D manifold of images
    figure = np.zeros((img_size * n, img_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
 
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sample = torch.tensor([[xi, yi]])
            x_decoded = vae.decode(sample, verbose=0)['x_recon'].squeeze(0)            
            images = x_decoded.permute(1, 2, 0).cpu().detach().numpy()

            figure[
                i * img_size : (i + 1) * img_size,
                j * img_size : (j + 1) * img_size,
            ] = images

    plt.figure(figsize=(figsize, figsize))
    start_range = img_size // 2
    end_range = n * img_size + start_range
    pixel_range = np.arange(start_range, end_range, img_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

def day_time(timezone='Asia/Ho_Chi_Minh'):
    now_utc = datetime.now(pytz.utc)

    timezone = pytz.timezone(timezone)
    now = now_utc.astimezone(timezone)

    formatted_date = now.strftime("%d-%m-%Y")
    formatted_time = now.strftime("%H-%M")

    return formatted_date, formatted_time    