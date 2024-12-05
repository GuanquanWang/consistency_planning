import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet2',
        'horizon': 16,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 2, 4, 8),
        'attention': True,
        'returns_condition': True,
        'dim': 64,
        'condition_dropout': 0.25,
        'calc_energy': False,
        'sigma_data': 0.5, 
        'sigma_max': 80.0,
        'sigma_min': 0.002,
        'distillation': False,
        'weight_schedule': 'karras',

        ## dataset

        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,
        'include_returns': True,
        # 'discount': 0.99,
        'returns_scale': 400.0,
        'termination_penalty': -100.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': '0.999,0.9999,0.9999432189950708',
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,

        'bucket': None,
        'device': 'cuda:0',
        'seed': 100,
    },

    'plan': {
        'ema': '0.999',
        'w': 0.,
        'test_ret': 0.9,
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 150,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 100,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',

        'diffusion_epoch': 'latest',

        'verbose': True,

        ##new
        'ts': [],
        'sampler': 'heun',
        'num_samples': 1,

        'steps': 40,
        'clip_denoised': True,
        'sigma_max': 80.0,
        'sigma_min': 0.002,
        's_churn': 0.0,
        's_tmin': 0.0,
        's_tmax': float("inf"),
        's_noise': 1.0,
        'generator': 'dummy',
        'progress': False,
        'callback': None,
    },

    'multistep':{
        # serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'multistep/',
        'diffusion_epoch': 'latest',
        'seed': 100,
        'exp_name': watch(args_to_watch),
        'use_changing_returns': False,

        'sampler': 'multistep',
        'generator': 'dummy',
        'test_ret': 0.9,
        'batch_size': 150,

        'horizon': 32,
        'steps': 40,
        'clip_denoised': True,
        'progress': False,
        'callback': None,
        'sigma_min': 0.002,
        'sigma_max': 80.0,
        's_churn': 0.0,
        's_tmin': 0.0,
        's_tmax': float("inf"),
        's_noise': 1.0,
        'device': 'cuda',

        'ts': '0,22,39',
        'configpath': 'openai-2024-09-23-14-05-53-556014',  #CD
        'model_place': '/data/local/diffuser/save_model/openai-2024-09-23-14-06-11-003587', #CD
    },

    'distill': {
        ## ema
        'target_ema_mode': 'fixed',
        'start_ema': 0.95,
        'scale_mode': 'fixed',
        'start_scales': 40,
        'end_scales': 40,
        'total_training_steps': 600000,
        'distill_steps_per_iter': 50000,

        'training_mode': 'consistency_distillation',
        'schedule_sampler': 'uniform',
        'use_fp16': False,
        'global_batch_size': 2048,
        'batch_size': -1,

        'horizon': 32,

        ## model
        'model': 'models.TemporalUnet2',
        'w_cond_dim': 512,
        'diffusion': 'models.KarrasDenoiser',
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'attention': True,
        'renderer': 'utils.MuJoCoRenderer',
        'returns_condition': True,
        'dim': 128,
        'condition_dropout': 0.25,
        'calc_energy': False,
        'sigma_data': 0.5, 
        'sigma_max': 80.0,
        'sigma_min': 0.002,
        'device': 'cuda',
        'weight_schedule': 'karras',

        ## loading
        'diffusion_epoch': 'latest',
        'seed': 100,
        'ema': '0.999',

        ## serialization

        'logbase': logbase,
        'prefix': 'distill/defaults',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## training
        'fp16_scale_growth': 0.001,
        'microbatch': -1,
        'lr': 0.000008,
        'ema_rate': '0.999,0.9999,0.9999432189950708',
        'log_interval': 10,
        'save_interval': 10000,
        'resume_checkpoint': '',
        'weight_decay': 0.0,
        'lr_anneal_steps': 0,
    }
}


#------------------------ overrides ------------------------#

halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'returns_scale': 1200.0,
        'dim': 128,
        'horizon': 32,
        'history_length': 8,
        'dim_mults': (1, 4, 8),
        'reward_shaping': True,
    },
}