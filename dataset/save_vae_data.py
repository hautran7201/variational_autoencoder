import sys
sys.path.append('.')

from utils import load_multi_dataset

# Train 
link = {
    'nerf_synthetic_data': {
        'dataset_path': 'path/nerf_synthetic',
        'splits': ['train'],
        'num_data': 1
    },
    'nerf_llff_data': {
        'dataset_path': 'path/nerf_llff_data',
        'splits': ['images_4'],
        'num_data': 1
    },
    'objaverse_eval_data': {
        'dataset_path': 'path/objaverse_eval_data/images',
        'splits': ['train'],
        'num_data': 1
    }
}

path = 'dataset/vae_dataset/train'
data = load_multi_dataset(data_structure=link, saving_path=path)

# Test
link = {
    'nerf_synthetic_data': {
        'dataset_path': 'path/nerf_synthetic',
        'splits': ['test'],
        'num_data': 1
    },
    'nerf_llff_data': {
        'dataset_path': 'path/nerf_llff_data',
        'splits': ['images_4'],
        'num_data': 1
    },
    'objaverse_eval_data': {
        'dataset_path': 'path/objaverse_eval_data/images',
        'splits': ['test'],
        'num_data': 1
    }
}

path = 'dataset/vae_dataset/test'
data = load_multi_dataset(data_structure=link, saving_path=path)