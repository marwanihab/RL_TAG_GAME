import torch
import torch.nn.functional as F
from ornsteinUhlenbeck import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grad_norm_clipping_critic = 0.5
grad_norm_clipping_actor = 0.5


class Agent:

    def __init__(self, pos, actor, critic, actor_target, critic_target, train_mode, discrete_action, args,
                 alg_mode='MADDPG'):

        self.pos = pos
        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.GAMMA
        self.args = args
        self.train_mode = train_mode
        self.discrete_action = discrete_action
        self.algorithm = alg_mode

        self.critic = critic
        self.critic_target = critic_target

        self.noise = OrnsteinUhlenbeckProcess(mu=np.zeros(5, ))

        self.actor = actor
        self.actor_target = actor_target

        self.actor_target.hard_copy(actor)
        self.critic_target.hard_copy(critic)

        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = self.BATCH_SIZE * 25

    def preupdate(self):
        self.replay_sample_index = None

    def step(self, agents, t, terminal):

        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.BATCH_SIZE)

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for agent in agents:
            obs, act, rew, obs_next, done = agent.replay_buffer.sample_index(index)
            obs_n.append(torch.FloatTensor(obs).to(device))
            obs_next_n.append(torch.FloatTensor(obs_next).to(device))
            act_n.append(torch.FloatTensor(act).to(device))

        state_batch, action_batch, reward_batch, state_next_batch, t_batch = self.replay_buffer.sample_index(index)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        t_batch = torch.FloatTensor(t_batch).to(device)
        state_next_batch = torch.FloatTensor(state_next_batch).to(device)
        reward_batch = torch.reshape(reward_batch, (1024, 1))
        t_batch = torch.reshape(t_batch, (1024, 1))

        # Train the critic network.
        if self.algorithm == 'MADDPG':
            if self.discrete_action:
                target_actions = [onehot_from_logits(agent.actor_target(nobs)) for agent, nobs in
                                  zip(agents, obs_next_n)]
            else:
                target_actions = [agent.actor_target(nobs) for agent, nobs in zip(agents, obs_next_n)]

            obs_next_concat = torch.cat(obs_next_n, dim=-1)
            target_actions = torch.cat(target_actions, dim=-1)
        else:  # Get actions in DDPG mode.
            if self.discrete_action:
                target_actions = onehot_from_logits(self.actor_target(state_next_batch))
            else:
                target_actions = self.actor_target(state_next_batch)
            obs_next_concat = state_next_batch

        predicted_q_value = self.critic_target(obs_next_concat, target_actions)
        Q_targets = reward_batch + ((1 - t_batch) * self.GAMMA * predicted_q_value).detach()

        if self.algorithm == 'MADDPG':
            obs_concat = torch.cat(obs_n, dim=-1)
            action_concat = torch.cat(act_n, dim=-1)
        else:
            obs_concat = state_batch
            action_concat = action_batch

        self.critic.train_step(obs_concat, action_concat, Q_targets)

        all_actions = []
        if self.discrete_action:
            curr_pol_out = self.actor(state_batch)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.actor(state_batch)
            curr_pol_vf_in = curr_pol_out

        if self.algorithm == 'MADDPG':  # Get the actions of all actors in MADDPG mode.
            for i, agent, obs in zip(range(len(agents)), agents, obs_n):
                if i == self.pos:
                    all_actions.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_actions.append(onehot_from_logits(agent.actor(obs)))
                else:
                    all_actions.append(agent.actor(obs))
            actions_concatenated = torch.cat(all_actions, dim=-1)
        else:  # Get ONLY the action of the current actor in DDPG.
            actions_concatenated = curr_pol_vf_in

        self.actor.train_step(self.critic, obs_concat, actions_concatenated, curr_pol_out)

        self.soft_update(self.actor, self.actor_target, tau=self.args.tau)
        self.soft_update(self.critic, self.critic_target, tau=self.args.tau)

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, done)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        noise = self.noise()
        noise = torch.FloatTensor(noise).unsqueeze(0).to(device)
        action = self.actor(state)

        if self.discrete_action:
            if add_noise:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:
            if add_noise:
                action = action + noise
            action = action.clamp(-1, 1)

        action = action.cpu().detach().numpy()[0]
        return action

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False).to(device)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

# training q network (critic)
# act_next = [F.softmax(agents[i].actor_target(obs_next_n[i])) for i in range(0, self.n)]
# # print(len(act_next))
# tensor_act_next = torch.cat(act_next, dim=-1)
# # print(tensor_act_next.size())
# #print(obs_next_n)
# tensor_obs_next_n = torch.cat(obs_next_n, dim=-1)
# #print(tensor_obs_next_n.size())
# #print(self.critic_target)
#
# q_targets_next = self.critic_target(tensor_obs_next_n, tensor_act_next)
# q_targets = torch.clamp(rew + (self.args.GAMMA * (1.0 - done) * q_targets_next), min=-1, max=1)
#
# tensor_obs_n = torch.cat(obs_n, dim=-1)
# tensor_act_n = torch.cat(act_n, dim=-1)
#
# q_expected = self.critic(tensor_obs_n, tensor_act_n)
# critical_loss = F.mse_loss(q_expected, q_targets)
# self.critic_optimizer.zero_grad()
# critical_loss.backward()
# clip_grad_norm(self.critic.parameters(), max_norm=grad_norm_clipping_critic)
# self.critic_optimizer.step()
# if self.alg_mode == 'MADDPG':
#     if self.discrete_action:
#         target_actions = [onehot_from_logits(agent.actor_target(nobs)) for agent, nobs in
#                           zip(agents, obs_next_n)]
#     else:
#         target_actions = [agent.actor_target(nobs) for agent, nobs in zip(agents, obs_next_n)]
#
#     obs2_concat = torch.cat(obs_next_n, dim=-1)
#     target_actions = torch.cat(target_actions, dim=-1)
# else:  # Get actions in DDPG mode.
#     if self.discrete_action:
#         target_actions = onehot_from_logits(self.actor_target(obs_next_batch))
#     else:
#         target_actions = self.actor_target(obs_next_batch)
#     obs2_concat = obs_next_batch
#
# predicted_q_value = self.critic_target(obs2_concat, target_actions)
# yi = rew + ((1 - done) * self.GAMMA * predicted_q_value).detach()
#
# if self.alg_mode == 'MADDPG':
#     obs_concat = torch.cat(obs_n, dim=-1)
#     action_concat = torch.cat(act_n, dim=-1)
# else:
#     obs_concat = obs_batch
#     action_concat = act
#
# predictions = self.critic.train_step(obs_concat, action_concat, yi)
#
# ep_ave_max_q_value = np.amax(predictions.cpu().detach().numpy())

# training p network (actor)
# action = self.actor(obs)
# new_act_n = act_n
# new_act_n[self.agent_index] = F.softmax(action)
#
# tensor_action = torch.cat(new_act_n, dim=-1)
# loss = -(self.critic(tensor_obs_n, tensor_action).mean())
# actor_loss = ((action.pow(2)).mean() * 1e-3) + loss
# self.actor_optimizer.zero_grad()
# actor_loss.backward()
# clip_grad_norm(self.actor.parameters(), max_norm=grad_norm_clipping_actor)
# self.actor_optimizer.step()

# all_pol_acs = []
# if self.discrete_action:
#     curr_pol_out = self.actor(obs_batch)
#     curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
# else:
#     curr_pol_out = self.actor(obs_batch)
#     curr_pol_vf_in = curr_pol_out
#
# if self.alg_mode == 'MADDPG':  # Get the actions of all actors in MADDPG mode.
#     for i, agent, obs in zip(range(len(agents)), agents, obs_n):
#         if i == self.pos:
#             all_pol_acs.append(curr_pol_vf_in)
#         elif self.discrete_action:
#             all_pol_acs.append(onehot_from_logits(agent.actor(obs)))
#         else:
#             all_pol_acs.append(agent.actor(obs))
#     act_n_concat = torch.cat(all_pol_acs, dim=-1)
# else:  # Get ONLY the action of the current actor in DDPG.
#     act_n_concat = curr_pol_vf_in
#
# self.actor.train_step(self.critic, obs_concat, act_n_concat, curr_pol_out)
