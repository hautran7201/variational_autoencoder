import os
import torch
import random
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def load_vae_data_from_disk(data_path='data/vae', shuffle=False):
    # Load data
    total_data = []
    for name in os.listdir(data_path):
        data = torch.load(os.path.join(data_path, name))

        for key, value in data.items():
            total_data.append(value['images'])

    # Shuffle data
    total_data = torch.cat(total_data, 0)
    if shuffle:
        shuffled_tensor = torch.randperm(total_data.size(0))
        total_data = total_data[shuffled_tensor]

    return total_data


def load_multi_dataset(data_structure, saving_path=None):
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
        data = load_dataset(
            dataset_path, 
            splits,
            num_data
        )

        datas[key] = data

        # Save data
        if saving_path:
            torch.save(data, os.path.join(saving_path, f'{key}.pt'))

    return datas


def load_dataset(
        dataset_path: str, 
        splits: list=[],
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
        if num_data:
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

            if (len(path_list) >= num_data) and num_data:
                break

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