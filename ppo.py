import numpy as np
from buffer import Buffer 
from actor_critic import ActorCritic
import torch.nn as nn
import torch
from datetime import datetime
import os
import matplotlib.pyplot as plt
from plotter import save_individual_plots, plot_training_curves

def get_one_hot(size, idx):
    oh = np.zeros(size)
    oh[idx - 1] = 1.0
    return oh



def ppo(env, actor_layers, critic_layers, iterations, epochs_per_iteration, num_steps_per_epoch, actor_lr, critic_lr, gamma, lamb, entropy_weight, ppo_clip):

    buffer = Buffer(num_steps_per_epoch, env.observation_space.n, 1, gamma, lamb) #action size of 1
    ac = ActorCritic(actor_layers, critic_layers, nn.ReLU)

    # Create log directory for plots
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = f"checkpoints/{timestamp}"
    plots_dir = f"plots/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
        
    f = open(f"{plots_dir}/log.txt", 'w')
    
    actor_optimizer = torch.optim.Adam(ac.actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(ac.v.parameters(), lr=critic_lr)

    policy_losses = []
    critic_losses = []
    returns = []
    entropy_losses = []
    iterations_list = []

    for iteration in range(iterations):
        iterations_list.append(iteration)
        obs, info = env.reset()
        
        for step in range(num_steps_per_epoch):
            # obs = get_one_hot(env.observation_space.n, obs)
            act, val, logp = ac.step(get_one_hot(env.observation_space.n, obs))
            # act = get_scalar_from_one_hot(act)
            act = act.item()
            
            obs_next, rew, terminated, truncated, info = env.step(act)
            if obs_next == 47:
                f.write("Won the game\n")
            # truncated always false since no max length set
            buffer.set(obs, act, rew, val, logp)
            if terminated:
                buffer.finish_path()  # Calculate advantages and returns
                obs, info = env.reset()

            if truncated:
                _, val, _ = ac.step(obs)
                buffer.finish_path(val)
                obs, info = env.reset()

            obs = obs_next
        
        _, val, _ = ac.step(get_one_hot(env.observation_space.n, obs))
        buffer.finish_path(val) 
        
        data = buffer.get() #in pytorch form

        for epoch in range(epochs_per_iteration):
            obs_batch, act_batch, rew_batch, ret_batch,  val_batch, logp_pi_old_batch, adv_batch = data['obs'], data['act'], data['rew'], data['ret'], data['val'], data['logp'], data['adv']
            # obs_batch: batch x obs_space
            # act_batch: batch x 1
            # logp_batch: batch x 1
            # adv_batch: batch x 1
            # optimize the actor and critic
            
            # adv batch normalization
            adv_batch = (adv_batch-adv_batch.mean())/(adv_batch.std() + 1e-8)
            print("number of non-zero advantages", adv_batch[adv_batch!= adv_batch.mean()].sum())


            act_disbn = ac.actor.get_distribution(ac.actor(obs_batch))
            logp_pi_batch = act_disbn.log_prob(act_batch)
            
            ratio = torch.exp(logp_pi_batch - logp_pi_old_batch)
            print('ratio is', ratio.mean())
            
            # TODO: add entropy bonus and make sure that you do this clamping per sample here
            # print('adv_batch', adv_batch.mean())
            # print('ratio', ratio.mean(), logp_pi_batch.mean(), logp_pi_old_batch.mean())
            # print('term 1', (adv_batch * torch.clamp(ratio, 1-ppo_clip, 1+ppo_clip)).mean())
            # print('term 2', (adv_batch * ratio).mean())
            
            loss_clip = -torch.min(adv_batch * torch.clamp(ratio, 1-ppo_clip, 1+ppo_clip), adv_batch * ratio).mean() # batch x ppo 
            loss_actor_entropy = -act_disbn.entropy().mean()
            loss_actor = loss_clip + entropy_weight * loss_actor_entropy 
            actor_optimizer.zero_grad()
            loss_actor.backward()
            actor_optimizer.step()

            # optimize the critic now
            # numpy stuff returned here - no gradients to backprop on ...

            vals = ac.v(obs_batch)
            ret_batch = torch.clamp(ret_batch, min=-50.0, max=0.0)
            loss_critic = ((vals - ret_batch)**2).mean()
            print('vals', vals[:10])
            print('ret', ret_batch[:10])
            critic_optimizer.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(ac.v.parameters(), max_norm=1.0)  # Add this line
            critic_optimizer.step()
            ac.save(actor_optimizer, critic_optimizer, checkpoint_dir, epoch)

        # Convert to Python float for easier plotting
        policy_losses.append(loss_clip.item())
        entropy_losses.append((entropy_weight*loss_actor_entropy).item())
        returns.append(ret_batch.mean().item())
        critic_losses.append(loss_critic.item())
        
        print('policy losses are', policy_losses)
        
        # Plot and save training curves during training (every iteration)
        # plot_training_curves(iterations_list, policy_losses, critic_losses, returns, entropy_losses, plots_dir, save=True)
        
        f.write(f"Iteration: {iteration}, Avg Return: {ret_batch.mean()}, Policy CLIP loss: {loss_clip}, Entropy Loss: {entropy_weight * loss_actor_entropy} Critic Loss: {loss_critic}\n")
    
    # After training is complete, create final plots with all data
    final_plots_dir = f"{plots_dir}/final"
    os.makedirs(final_plots_dir, exist_ok=True)
    plot_training_curves(iterations_list, policy_losses, critic_losses, returns, entropy_losses, final_plots_dir, save=True)
    
    f.close()
    return policy_losses, critic_losses, returns, entropy_losses
