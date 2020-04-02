import torch
import torch.nn
from agent import Agent
import numpy as np
import time
import argparse
import pickle
from gym.spaces import Box
from actor_critic_model import Actor, Critic

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument('--scenario', type=str, default='simple_tag')
    parser.add_argument("--max_episode_len", type=int, default=36, help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num_adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good_policy", type=str, default="MADDPG", help="policy for good agents")
    parser.add_argument("--adv_policy", type=str, default="DDPG", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--GAMMA", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--tau", type=float, default=0.01, help="TAU for the soft update of the network")
    parser.add_argument('--eval', action='store_false')

    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    # parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
    #                     help="directory where plot data is saved")
    # print(parser.parse_args())
    return parser.parse_args()


def make_env(scenario_name, arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(arglist.num_adversaries)
    # create multiagent environment
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
    return env


def _algo_mode_from_agents(env, arglist):
    algo_mode = []

    for agent in env.agents:
        if agent.adversary:  # adversary
            algo_mode.append(arglist.adv_policy)  # MADDPG
        else:
            algo_mode.append(arglist.good_policy)
    return algo_mode


def create_agents(env, arglist):
    agents = []
    algo_mode = _algo_mode_from_agents(env=env, arglist=arglist)

    obs_shapes = [env.observation_space[i].shape for i in range(env.n)]
    actions_shape_n = [env.action_space[i].n for i in range(env.n)]
    actions_n = 0
    obs_shape_n = 0

    for actions in actions_shape_n:
        actions_n += actions
    for obs_shape in obs_shapes:
        obs_shape_n += obs_shape[0]

    for i, action_space, observation_space, algo in zip(range(len(env.action_space)), env.action_space,
                                                        env.observation_space, algo_mode):

        if isinstance(action_space, Box):
            discrete_action = False
        else:
            discrete_action = True

        if algo == 'MADDPG':
            print('MADDPG load.')
            critic = Critic(obs_shape_n, actions_n).to(device)
            actor = Actor(observation_space.shape[0], action_space.n).to(device)
            target_critic = Critic(obs_shape_n, actions_n, arglist.tau).to(device)
            target_actor = Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)
        else:
            print('DDPG load.')
            critic = Critic(observation_space.shape[0], action_space.n).to(device)
            actor = Actor(observation_space.shape[0], action_space.n).to(device)
            target_critic = Critic(observation_space.shape[0], action_space.n, arglist.tau).to(device)
            target_actor = Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)
        actor.eval()
        critic.eval()
        target_actor.eval()
        target_critic.eval()
        agents.append(
            Agent(i, actor, critic, target_actor, target_critic, arglist.eval, discrete_action, arglist, algo))
    return agents


def train(arglist):
    env = make_env(scenario_name="simple_tag", arglist=arglist)
    # ACTORS = 1
    # env = EnvWrapper(arglist.scenario, ACTORS, arglist.saved_episode)
    agents = create_agents(env, arglist)
    max_episode_len = arglist.max_episode_len

    if arglist.display:
        for i in range(len(agents)):
            actor = agents[i].actor
            actor_ckpt = torch.load('./checkpoints_{}_{}_{}/checkpoint_actor_{}.pth'.format(arglist.good_policy,
                                                                                            arglist.adv_policy,
                                                                                            arglist.num_adversaries, i),
                                    map_location='cpu')
            actor.load_state_dict(actor_ckpt)
            actor_target = agents[i].actor_target
            actor_target_ckpt = torch.load(
                './checkpoints_{}_{}_{}/checkpoint_actor_target_{}.pth'.format(arglist.good_policy,
                                                                               arglist.adv_policy,
                                                                               arglist.num_adversaries, i),
                map_location='cpu')
            actor_target.load_state_dict(actor_target_ckpt)
            critic = agents[i].critic
            critic_ckpt = torch.load('./checkpoints_{}_{}_{}/checkpoint_critic_{}.pth'.format(arglist.good_policy,
                                                                                              arglist.adv_policy,
                                                                                              arglist.num_adversaries,
                                                                                              i),
                                     map_location='cpu')
            critic.load_state_dict(critic_ckpt)
            critic_target = agents[i].critic_target
            critic_target_ckpt = torch.load(
                './checkpoints_{}_{}_{}/checkpoint_critic_target_{}.pth'.format(arglist.good_policy,
                                                                                arglist.adv_policy,
                                                                                arglist.num_adversaries, i),
                map_location='cpu')
            critic_target.load_state_dict(critic_target_ckpt)

    final_ep_rewards = []
    final_ep_ag_rewards = []
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]
    agent_info = [[[]]]
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        for agent in agents:
            agent.reset()

        action_n = [agent.act(obs, add_noise=False) for agent, obs in zip(agents, obs_n)]

        # environment step
        # print(len(action_n))
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= max_episode_len)
        # collect experience
        for i, agent in enumerate(agents):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
        obs_n = new_obs_n
        # print(rew_n)
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            # print(episode_rewards)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        if done:
            print("\n\n###############################################\n"
                  "ADVERSARY TAGGED THE GOOD AGENT, EPISODE ENDED\n"
                  "#############################################\n\n")

        # increment global step counter
        train_step += 1
        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal) and (len(episode_rewards) % 1000 == 0):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                # break
            # continue
        if terminal:
            print("\n\n************************************************************\n"
                  "TIME ELAPSED, GOOD AGENT WON THE EPISODE WITHOUT BEING TAGGED\n"
                  "****************************************************************\n\n")
        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in agents:
            agent.preupdate()
        for agent in agents:
            loss = agent.step(agents, train_step, terminal)

        # save model, display training output
        if terminal and (len(episode_rewards) % 1000 == 0):  # 25 and 1000

            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-1000:]), round(time.time() - t_start, 3)))
            i = 0
            for agt in agents:
                torch.save(agt.actor.state_dict(),
                           './checkpoints_{}_{}_{}/checkpoint_actor_{}.pth'.format(arglist.good_policy,
                                                                                   arglist.adv_policy,
                                                                                   arglist.num_adversaries, i))
                torch.save(agt.actor_target.state_dict(),
                           './checkpoints_{}_{}_{}/checkpoint_actor_target_{}.pth'.format(arglist.good_policy,
                                                                                          arglist.adv_policy,
                                                                                          arglist.num_adversaries, i))
                torch.save(agt.critic.state_dict(),
                           './checkpoints_{}_{}_{}/checkpoint_critic_{}.pth'.format(arglist.good_policy,
                                                                                    arglist.adv_policy,
                                                                                    arglist.num_adversaries,
                                                                                    i))
                torch.save(agt.critic_target.state_dict(),
                           './checkpoints_{}_{}_{}/checkpoint_critic_target_{}.pth'.format(arglist.good_policy,
                                                                                           arglist.adv_policy,
                                                                                           arglist.num_adversaries, i))

                i += 1

            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-1000:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-1000:]))

        if len(episode_rewards) > arglist.num_episodes:
            break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist=arglist)
