import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset model diffusion')
def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath, ema):
    states = glob.glob1(os.path.join(*loadpath), 'ema_'+ema+'_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('ema_'+ema+'_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0', seed=None, model_place=None, ema=None, invdyn=False):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')


    dataset = dataset_config(seed=seed)
    model = model_config()
    diffusion = diffusion_config()
    if epoch == 'latest':
        epoch = get_latest_epoch((model_place,), ema)
        epoch = str(epoch).zfill(6)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}, ema: {ema}\n')


    loadpath = os.path.join(model_place, f'ema_{ema}_{epoch}.pt')
    model.load_state_dict(torch.load(loadpath))
    model.to(device)
    model.eval()
    return DiffusionExperiment(dataset, model, diffusion)

def check_compatibility(experiment_1, experiment_2):
    '''
        returns True if `experiment_1 and `experiment_2` have
        the same normalizers and number of diffusion steps
    '''
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        assert norm_1 == norm_2, \
            f'Normalizers should be identical, found {norm_1} and {norm_2} for field {key}'

    n_steps_1 = experiment_1.diffusion.n_timesteps
    n_steps_2 = experiment_2.diffusion.n_timesteps
    assert n_steps_1 == n_steps_2, \
        ('Number of timesteps should match between diffusion experiments, '
        f'found {n_steps_1} and {n_steps_2}')
