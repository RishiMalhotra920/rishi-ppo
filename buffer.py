import numpy as np
import torch 

class Buffer:
    def __init__(self, size, s_size, a_size, gamma, lamb):
        self.size = size
        self.ptr = 0 # invariant: always points to the next open slot
        self.obs_buf = np.zeros((size, s_size))
        self.act_buf = np.zeros((size, a_size))
        self.val_buf = np.zeros(size)
        self.rew_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.adv_buf = np.zeros(size)
        self.logp_buf = np.zeros(size)
        self.path_start_idx = 0
        self.gamma, self.lamb = gamma, lamb

    def set(self, obs, act, rew, val, logp):
        # idea is that you set these values as you get them
        # and then when the trajectory is done, call finish_path
        # which computes the ret and adv
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act 
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    

    def _cum_sum(self, arr, factor):
        result = np.zeros_like(arr, dtype=float)
        running = 0
        
        for i in reversed(range(len(arr))):
            running = arr[i] + factor * running
            result[i] = running
            
        return result 

    def finish_path(self, last_val=0):
        """calculate the returns and adv and prepare buffer to 
        collect new trajectories"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.ret_buf[path_slice] = self._cum_sum(rews, self.gamma)[:-1]

        delta_i = rews[:-1] + self.gamma*vals[1:] - vals[:-1] # R + gamma * V(S') - V(S)
        advs = self._cum_sum(delta_i, self.gamma * self.lamb) #GAE
        self.adv_buf[path_slice] = (advs-advs.mean())/(advs.std() + 1e-8)

        self.path_start_idx = self.ptr
        

    def get(self):
        """Extract data and reset the buffer"""
        d = dict(obs=self.obs_buf[:self.ptr],
                act=self.act_buf[:self.ptr].squeeze(),
                val=self.val_buf[:self.ptr],
                rew=self.rew_buf[:self.ptr],
                ret=self.ret_buf[:self.ptr],
                adv=self.adv_buf[:self.ptr],
                logp=self.logp_buf[:self.ptr])
        
        self.ptr = 0
        self.path_start_idx = 0
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}



