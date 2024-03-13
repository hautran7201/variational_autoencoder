from utils import load_multi_dataset

link = {
    'nerf_synthetic_data': {
        'dataset_path': 'nerf_synthetic',
        'splits': ['train'],
        'num_data': 20
    },
    'nerf_llff_data': {
        'dataset_path': 'nerf_llff_data',
        'splits': ['images_4'],
        'num_data': 20
    },
    'nerf_llff_data': {
        'dataset_path': 'objaverse_eval_data/images',
        'splits': ['train'],
        'num_data': 10
    }
}

path = 'data/vae'
load_multi_dataset(data_structure=link, saving_path=path)