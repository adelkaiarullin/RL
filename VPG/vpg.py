import torch
from torch import nn
import gym


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(4, 18), nn.SELU()
                    ,nn.Linear(18, 18), nn.SELU(), nn.Linear(18, 2), nn.Softmax())

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    agent = AgentNet()
    optimizer_ag = torch.optim.Adam(agent.parameters(), lr=1e-3)
    agent.to(device)
    
    env = gym.make('CartPole-v1')

    #env = gym.make('YarsRevenge-ramNoFrameskip-v0')
    print(env.action_space)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    # print(gym.envs.registry.all())

    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(1000):
    #         env.render()
    #         #print(observation)
    #         action = env.action_space.sample()
    #         #print(index, pred, pred[index])
    #         observation, reward, done, info = env.step(action)
            
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    # exit(1)

    history_r = []
    history_l = []
    history_s = []
    logprob = 0
    mse = 0
    batch_size = 10
    counter = 0

    for i_episode in range(2000):
        print('Episode {}'.format(i_episode))
        R = 0
        Rtg = []
        LogProb = []
        observation = env.reset()
        for t in range(200):
            #env.render()
            #action = env.action_space.sample()
            obs = torch.tensor(observation.reshape(1, 4), dtype=torch.float32, device=device)
            pred = agent.forward(obs)[0]
            index = torch.multinomial(pred, 1)[0]
            # print(index, pred, pred[index])
            observation, reward, done, info = env.step(index.cpu().numpy())

            R += reward
            Rtg.append(R)
            LogProb.append(-torch.log(pred[index]))
            
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                if counter % batch_size == 0:
                    history_s.append(t)
                    history_r.append(R)
                counter += 1
                break

        Rtg.reverse()
        for p, r in zip(LogProb, Rtg):
            logprob += r * p

        if counter % batch_size == 0:
            logprob /= batch_size
            optimizer_ag.zero_grad()
            logprob.backward()
            optimizer_ag.step()
            history_l.append(logprob.item())
            mse = 0
            logprob = 0
            LogProb.clear()
            Rtg.clear()

    agent.eval()
    torch.save(agent, 'rlnet.pt')

    for i_episode in range(20):
        observation = env.reset()
        for t in range(200):
            env.render()
            # action = env.action_space.sample()
            obs = torch.tensor(observation.reshape(1, 4), dtype=torch.float32, device=device)
            pred = agent.forward(obs)[0]
            index = torch.multinomial(pred, 1)[0]
            #print(index, pred, pred[index])
            observation, reward, done, info = env.step(index.cpu().numpy())
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break      

    env.close()

    import matplotlib.pyplot as plt
    plt.subplot('131')
    plt.title('Step')
    plt.plot(range(len(history_s)), history_s)
    plt.subplot('132')
    plt.title('Reward')
    plt.plot(range(len(history_r)), history_r)
    plt.subplot('133')
    plt.title('Loss')
    plt.plot(range(len(history_l)), history_l)
    plt.show()
