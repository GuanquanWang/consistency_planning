import diffuser.utils as utils
from diffuser.models import dist_util
from diffuser.models.resample import create_named_schedule_sampler
from diffuser.models.script_util import create_ema_and_scales_fn
from diffuser.models.train_util import CMTrainLoop
import torch.distributed as dist
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('distill')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.model_place, model_place=args.model_place,
    epoch=args.diffusion_epoch, seed=args.seed, ema=args.ema,
)

teacher_diffusion = diffusion_experiment.diffusion
dataset = diffusion_experiment.dataset
teacher_model = diffusion_experiment.model

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

dist_util.setup_dist()
ema_scale_fn = create_ema_and_scales_fn(
    target_ema_mode=args.target_ema_mode,
    start_ema=args.start_ema,
    scale_mode=args.scale_mode,
    start_scales=args.start_scales,
    end_scales=args.end_scales,
    total_steps=args.total_training_steps,
    distill_steps_per_iter=args.distill_steps_per_iter,
)
if args.training_mode == "progdist":
    distillation = False
elif "consistency" in args.training_mode:
    distillation = True
else:
    raise ValueError(f"unknown training mode {args.training_mode}")

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon+dataset.history_length,
    transition_dim=observation_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    returns_condition=args.returns_condition,
    dim=args.dim,
    condition_dropout=args.condition_dropout,
    calc_energy=args.calc_energy,
    attention=args.attention,
    device=args.device,
    w_cond_dim=args.w_cond_dim,
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
    distillation=distillation,
    weight_schedule=args.weight_schedule,
    current_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    history_length=dataset.history_length,
    train_only_inv=False,
)

model = model_config()
diffusion = diffusion_config()

model.to(dist_util.dev())
model.train()

schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

if args.batch_size == -1:
    batch_size = args.global_batch_size // dist.get_world_size()
else:
    batch_size = args.batch_size

list_parameters = list(model.named_parameters())
for named_parameter in model.named_parameters():
    if 'w_mlp' in named_parameter[0]:
        list_parameters.remove(named_parameter)

for dst, src in zip(list_parameters, teacher_model.named_parameters()):
    assert dst[0] == src[0]
    dst[1].data.copy_(src[1].data)

target_model = model_config()


target_model.to(dist_util.dev())
target_model.train()

dist_util.sync_params(target_model.parameters())
dist_util.sync_params(target_model.buffers())

for dst, src in zip(target_model.parameters(), model.parameters()):
    dst.data.copy_(src.data)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

CMTrainLoop(
    model=model,
    target_model=target_model,
    teacher_model=teacher_model,
    teacher_diffusion=teacher_diffusion,
    training_mode=args.training_mode,
    ema_scale_fn=ema_scale_fn,
    total_training_steps=args.total_training_steps,
    diffusion=diffusion,
    data=dataset,
    batch_size=batch_size,
    microbatch=args.microbatch,
    lr=args.lr,
    ema_rate=args.ema_rate,
    log_interval=args.log_interval,
    save_interval=args.save_interval,
    resume_checkpoint=args.resume_checkpoint,
    use_fp16=args.use_fp16,
    fp16_scale_growth=args.fp16_scale_growth,
    schedule_sampler=schedule_sampler,
    weight_decay=args.weight_decay,
    lr_anneal_steps=args.lr_anneal_steps,
).run_loop()
