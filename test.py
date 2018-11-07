import time

import numpy as np
import torch
import torch.nn.functional as F

from visdom import Visdom
import subprocess
from environments import create_env
from model import ActorCritic


def test(rank, args, shared_model, counter, training_num):
    torch.manual_seed(args.seed + rank)

    env = create_env(args, 12500, False, 0)

    model = ActorCritic(1, 7)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    with open("/dev/null", "w") as out:
        cmd = ["python3", "-m", "visdom.server"]
        vis_server = subprocess.Popen(cmd, stdout=out, stderr=out)
        time.sleep(5)

        reward_vis = Visdom(env="reward")
        mean_reward_vis = Visdom(env="mean_reward")
        reward_win = reward_vis.line(X=torch.Tensor([0]), Y=torch.Tensor([0]))
        mean_reward_win = mean_reward_vis.line(X=torch.Tensor([0]), Y=torch.Tensor([0]))

        assert reward_vis.check_connection()
        assert mean_reward_vis.check_connection() 

    # a quick hack to prevent the agent from stucking
    # actions = deque(maxlen=100)
    max_episode_reward = -np.inf
    max_average_reward = -np.inf
    max_episode_length = 0
    recent_episode_reward = np.zeros(shape=[5])

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # actions.append(action[0, 0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     done = True

        if done:
            recent_episode_reward = np.roll(recent_episode_reward, 1)
            recent_episode_reward[0] = reward_sum
            state_to_save = model.state_dict()
            if max_episode_reward < reward_sum:
                max_episode_reward = reward_sum
                max_episode_length = episode_length
                torch.save(state_to_save, '{0}{1}-max.dat'.format(args.save_model_dir, args.env))

            if np.mean(recent_episode_reward) > max_average_reward:
                max_average_reward = np.mean(recent_episode_reward)
                torch.save(state_to_save, '{0}{1}-mean.dat'.format(args.save_model_dir, args.env))

            reward_sum = 0
            episode_length = 0
            # actions.clear()
            reward_win = reward_vis.line(Y=torch.Tensor([reward_sum]), X=torch.Tensor([training_num.value * args.num_steps]), win=reward_win, update="append")
            mean_reward_win = mean_reward_vis.line(Y=torch.Tensor([np.mean(recent_episode_reward)]), X=torch.Tensor([training_num.value * args.num_steps]), win=mean_reward_win, update="append")

            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, max episode reward {}, max episode length {} max average reward {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length, max_episode_reward, max_episode_length, max_average_reward))

            if training_num.value >= args.max_training_num:
                break

            state = env.reset()
            time.sleep(30)

        state = torch.from_numpy(state)

    env.end()
