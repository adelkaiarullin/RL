#the script writes video to show random sampling and agent's sampling
import cv2 as cv, numpy as np
import torch
import gym
from vpg import AgentNet


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = torch.load('rlnet.pt', map_location='cpu')
    device = torch.device('cpu')

    writer = cv.VideoWriter_fourcc('M','J','P','G')
    frame_width,frame_height = 600, 400
    fps = 10
    out = cv.VideoWriter('outpy.mp4', writer, fps , (frame_width,frame_height))
    font = cv.FONT_HERSHEY_SIMPLEX

    #random sampling
    for i_episode in range(20):
        observation = env.reset()
        for t in range(200):
            window = env.render(mode='rgb_array')
            window = window.copy()
            cv.putText(window, 'Random sampling. Step {} '.format(t+1), (5, 50), font, 1, (0,255,0),1, cv.LINE_AA)
            out.write(window)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    #sampling from a net.
    for i_episode in range(2):
        observation = env.reset()
        for t in range(200):
            window = env.render(mode='rgb_array')
            window = window.copy()
            cv.putText(window, 'Sampling from ANN. Step {} '.format(t+1), (5, 50), font, 1, (255,0,0),1, cv.LINE_AA)
            out.write(window)
            obs = torch.tensor(observation.reshape(1, 4), dtype=torch.float32, device=device)
            pred = net.forward(obs)[0]
            index = torch.multinomial(pred, 1)[0]
            observation, reward, done, info = env.step(index.numpy())
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break    
    
    env.close()
