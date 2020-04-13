import torch
from torch import nn
import gym


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(4, 9), nn.SELU(),
                                    nn.Linear(9, 9), nn.SELU(),
                                    nn.Linear(9, 9), nn.SELU(), 
                                    nn.Linear(9, 2), nn.Softmax())

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    agent = AgentNet()
    optimizer_ag = torch.optim.RMSprop(agent.parameters(), lr=1e-3)
    agent.to(device)
    
    env = gym.make('CartPole-v1')

    print(env.action_space)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)

    history_r = []
    logprob = 0

    for i_episode in range(1200):
        print('Episode {}'.format(i_episode))
        R = 0
        Rtg = []
        LogProb = []
        observation = env.reset()
        for t in range(200):
            obs = torch.tensor(observation.reshape(1, 4), dtype=torch.float32, device=device)
            pred = agent.forward(obs)[0]
            index = torch.multinomial(pred, 1)[0]
            observation, reward, done, info = env.step(index.cpu().numpy())

            R += reward
            Rtg.append(R)
            LogProb.append(-torch.log(pred[index]))
            
            if done:
                history_r.append(R)
                break

        Rtg.reverse()
        for p, r in zip(LogProb, Rtg):
            logprob += p * r

        logprob /= len(LogProb)
        optimizer_ag.zero_grad()
        logprob.backward()
        optimizer_ag.step()
        logprob = 0

    agent.eval()
    torch.save(agent, 'rlnet.pt')

    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(200):
    #         env.render()
    #         obs = torch.tensor(observation.reshape(1, 4), dtype=torch.float32, device=device)
    #         pred = agent.forward(obs)[0]
    #         index = torch.multinomial(pred, 1)[0]
    #         observation, reward, done, info = env.step(index.cpu().numpy())
            
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break      

    env.close()

    import matplotlib.pyplot as plt
    plt.title('Reward')
    plt.plot(history_r)
    plt.show()
