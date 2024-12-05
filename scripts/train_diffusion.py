import diffuser.utils as utils
from diffuser.models.resample import create_named_schedule_sampler
from diffuser.models.dist_util import setup_dist

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

setup_dist()
class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    'datasets.CondSequenceDataset',
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    include_returns=args.include_returns,
    discount=0.99,
    returns_scale=args.returns_scale, 
    termination_penalty=args.termination_penalty,
    history_length=args.history_length,
    reward_shaping=args.reward_shaping,
)

dataset = dataset_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon+args.history_length,
    transition_dim=observation_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    returns_condition=args.returns_condition,
    dim=args.dim,
    condition_dropout=args.condition_dropout,
    calc_energy=args.calc_energy,
    device=args.device,
    attention=args.attention,
)

diffusion_config = utils.Config(
    'models.KarrasInvDynDenoiser',
    savepath=(args.savepath, 
              'diffusion_config.pkl'),
    action_dim=action_dim,
    observation_dim=observation_dim,
    horizon=args.horizon,
    sigma_data=args.sigma_data,
    sigma_max=args.sigma_max,
    sigma_min=args.sigma_min,
    distillation=args.distillation,
    weight_schedule=args.weight_schedule,
    current_weight=1,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    history_length=dataset.history_length,
    train_only_inv=False,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config()
schedule_sampler = create_named_schedule_sampler('lognormal', diffusion)
trainer_config = utils.Config(
    'models.TrainLoop',
    savepath=(args.savepath, 'trainer_config.pkl'),
    batch_size=args.batch_size,
    microbatch=-1,
    lr=args.learning_rate,
    ema_rate=args.ema_decay,
    log_interval=10,
    save_interval=10000,
    resume_checkpoint='',
    use_fp16=False,
    fp16_scale_growth=0.001,
    schedule_sampler=schedule_sampler,
    weight_decay=0.0,
    lr_anneal_steps=0,
    use_history=False,
)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

trainer = trainer_config(model, diffusion, dataset,)
trainer.run_loop()
