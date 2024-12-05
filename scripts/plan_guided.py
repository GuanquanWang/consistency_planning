from diffuser.models.random_util import get_generator
from functools import partial
import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.models.karras_invdyn_diffusion import karras_sample
from diffuser.models import dist_util
import numpy as np
import torch as th
import pickle, gym

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dist_util.setup_dist()
class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('multistep')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.model_place,
    epoch=args.diffusion_epoch, seed=args.seed, model_place=args.model_place, ema=args.ema, invdyn=False,
)
print(f'w: {args.omega}')
inv_model = th.load(args.inv_model_path)
inv_model.to('cuda:0')
inv_model.eval()
diffusion = diffusion_experiment.diffusion
dataset = diffusion_experiment.dataset
model = diffusion_experiment.model

model.to(dist_util.dev())
model.eval()

if args.sampler == "multistep":
    assert len(args.ts) > 0
    ts = tuple(int(x) for x in args.ts.split(","))
else:
    ts = None
generator = get_generator(generator=args.generator, num_samples=args.batch_size, seed=args.seed)

returns = to_torch(np.array([args.test_ret] * args.batch_size)[:, np.newaxis])
image_length = dataset.horizon + dataset.history_length if dataset.history_length > 0 else dataset.horizon
sample =partial(karras_sample,
                diffusion=diffusion,
                model=model,
                shape=(args.batch_size, image_length, dataset.observation_dim),
                steps=args.steps,
                progress = args.progress,
                callback = args.callback,
                device=args.device,
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
                action_dim=dataset.action_dim,
                # w=args.w,
)
#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env_list = [gym.make(args.dataset) for _ in range(args.batch_size)]
dones = [0 for _ in range(args.batch_size)]
episode_rewards = [0 for _ in range(args.batch_size)]
scores = [0 for _ in range(args.batch_size)]
t = 0
obs_list = [env.reset(seed=index)[None] for index, env in enumerate(env_list)]
obs = np.concatenate(obs_list, axis=0)

total_reward = 0
if dataset.history_length > 0:
    conditions = to_torch({i: np.zeros([args.batch_size, dataset.observation_dim]) for i in range(dataset.history_length + 1)})
returns_collect = np.zeros([args.batch_size, 1000])


while sum(dones) <  args.batch_size:
    # format current observation for conditioning
    obs = dataset.normalizer.normalize(obs, 'observations')
    if dataset.history_length > 0:
        for i in range(dataset.history_length):
            conditions[i] = conditions[i+1].clone()
        conditions[dataset.history_length] = to_torch(obs)
    else:
        conditions = to_torch({0: obs})


        returns_collect[:, t] = returns.cpu().numpy().squeeze()
    with th.no_grad():
        if args.sampler == "multistep":
            samples = sample(model_kwargs=conditions, returns=returns, omega=[args.omega,] * args.batch_size)
        else:
            samples = sample(model_kwargs=conditions, returns=returns)
    obs_comb = th.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1) if dataset.history_length <= 0 else th.cat([samples[:, dataset.history_length, :], samples[:, dataset.history_length+1, :]], dim=-1)
    obs_comb = obs_comb.reshape(-1, 2*dataset.observation_dim)
    with th.no_grad():
        actions = inv_model(obs_comb)
    actions = to_np(actions.cpu())

    action = dataset.normalizer.unnormalize(actions, 'actions')

    obs_list = []
    print(f't: {t},   return: {returns[0].item():.4f}')
    for i in range(args.batch_size):
        ## execute action in environment
        print(action[i],'action')
        this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
        obs_list.append(this_obs[None])
        if this_done:
            if dones[i] == 1:
                this_reward = 0
            else:
                dones[i] = 1
                episode_rewards[i] += this_reward
                print(f"Episode ({i}): {episode_rewards[i]}")
        else:
            if dones[i] == 1:
                this_reward = 0
            else:
                episode_rewards[i] += this_reward
                
        scores[i] = env_list[i].get_normalized_score(episode_rewards[i])

    print(f"average_ep_reward: {np.mean(episode_rewards)}, mean_scr: {np.mean(scores)}")
    obs = np.concatenate(obs_list, axis=0)
    t += 1


episode_rewards = np.array(episode_rewards)
print(episode_rewards)
print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}, mean_scr: {env_list[0].get_normalized_score(np.mean(episode_rewards))}")
