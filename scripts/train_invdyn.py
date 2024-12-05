import diffuser.utils as utils
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import blobfile as bf
from diffuser.models import logger
import pickle
from diffuser.datasets import CondSequenceDataset, RewardBatch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

isval = False
batch_size = 1024
class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')

class FasterDataset(CondSequenceDataset):
    def __init__(self, env='hopper-medium-replay', horizon=64, seed=None, history_length=0, normalizer='LimitsNormalizer', 
                 preprocess_fns=..., max_path_length=1000, reward_shaping=False, max_n_episodes=10000, termination_penalty=-100, 
                 use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        super().__init__(env, horizon, seed, history_length, normalizer, preprocess_fns, max_path_length, 
                         reward_shaping, max_n_episodes, termination_penalty, use_padding, discount, returns_scale, include_returns)
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, returns, path_ind, start)
        else:
            batch = trajectories
        return batch

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    FasterDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=2,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=False,
    max_path_length=1000,
    include_returns=False,
    discount=0.99,#args.discount, 
    returns_scale=args.returns_scale, 
    termination_penalty=-100.0,
    history_length=0,
    reward_shaping=False,
)

data_medium_replay = FasterDataset(
    env='halfcheetah-medium-replay-v2',
    horizon=2,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=False,
    max_path_length=1000,
    include_returns=False,
    discount=0.99,
    returns_scale=args.returns_scale, 
    termination_penalty=-100.0,
    history_length=0,
    reward_shaping=False,
)

def train(rank, world_size, epochs):

    th.manual_seed(42)

    dataset_1 = dataset_config()
    dataset = ConcatDataset([dataset_1, data_medium_replay])

    if isval:
        n_samples = len(dataset)
        train_size = int(len(dataset) * 0.9)
        val_size = n_samples - train_size
        train_dataset, val_dataset = th.utils.data.random_split(dataset, [train_size, val_size])

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = th.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    else:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)

    observation_dim = dataset_1.observation_dim
    action_dim = dataset_1.action_dim
    hidden_dim = 256

    model = nn.Sequential(
                    nn.Linear(2 * observation_dim, hidden_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_dim * 2, hidden_dim*2),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(hidden_dim*2, action_dim),
                ).to(rank)

    if rank == 0: print(model)
    model = DDP(model, device_ids=[rank])
    optimizer = th.optim.Adam(model.parameters(), lr=1e-5)

    def fast_loss(x_start):
        obs_comb_t = x_start[:, :, action_dim:].reshape(-1, 2 * observation_dim)
        act_t = x_start[:, :-1, :action_dim].reshape(-1, action_dim)
        return obs_comb_t, act_t

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    record = {'epoch': [], 'loss': [], 'loss_val': []}

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = 0.0

        for image in train_loader:
            obs_comb_t, act_t = fast_loss(image.to(rank))
            optimizer.zero_grad()
            pred_a_t = model(obs_comb_t)
            loss = F.mse_loss(pred_a_t, act_t)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if (epoch + 1) % 100 == 0:

            total_train_loss_tensor = th.tensor(total_train_loss).to(rank)
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.barrier()  
            average_train_loss = total_train_loss_tensor.item() / (world_size * len(train_loader))
            if isval:
                model.eval()
                total_val_loss = 0.0
                with th.no_grad():
                    for val_img in val_loader:
                        obs_val, act_val = fast_loss(val_img.to(rank))
                        pred_val = model(obs_val)
                        loss_val = F.mse_loss(pred_val, act_val)
                        total_val_loss += loss_val.item()
            
                total_val_loss_tensor = th.tensor(total_val_loss).to(rank)
                dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.barrier() 
                average_val_loss = total_val_loss_tensor.item() / (world_size * len(val_loader))

            if rank == 0: 
                if isval:
                    print(f'Rank {rank}, Epoch {epoch+1}, Training Loss: {average_train_loss:5f}, Validation Loss: {average_val_loss:5f}')
                else:
                    print(f'Rank {rank}, Epoch {epoch+1}, Training Loss: {average_train_loss:5f},')
            record['epoch'].append(epoch+1)
            record['loss'].append(average_train_loss)
            if isval: record['loss_val'].append(average_val_loss)

        if (epoch + 1) % 100 == 0 and rank == 0: 
            filename = f"{args.dataset}_{(epoch + 1)}.pt"
            with bf.BlobFile(bf.join(logger.get_dir(), filename), "wb") as f:
                th.save(model.module, f)

    if rank == 0:
        recordname = f"{args.dataset}_{isval}_{(epoch + 1)}.pkl"
        with bf.BlobFile(bf.join(logger.get_dir(), recordname), "wb") as f:
            pickle.dump(record, f)

    cleanup()

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    epochs = 40000
    # train(epochs)
    train(rank, world_size, epochs)

if __name__ == "__main__":
    main()
