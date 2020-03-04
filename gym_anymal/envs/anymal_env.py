import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import numpy as np

import random

import math

import time

class AnymalEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, nb_steps=1000, gui=False):
        try:
            p.disconnect()
        except:
            pass
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotStartPos = [0,0,0.5]
        self.robotStartOrientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        self.robotId = p.loadURDF("laikago/laikago.urdf", self.robotStartPos, self.robotStartOrientation)
        self.joints = {}
        for i in range(p.getNumJoints(self.robotId)):
            self.joints[i] = p.getJointInfo(self.robotId, i)
        self.episode_over = False
        randomOffsetPos = [
            random.random()*10,
            random.random()*10,
            0
        ]
        #self.robotTargetPos = [sum(x) for x in zip(self.robotStartPos, randomOffsetPos)]
        #self.robotTargetOrientation = p.getQuaternionFromEuler([0,0,random.random()])
        self.current_steps = 0
        self.action_space = spaces.Box(0, 1, shape=[len(self.joints.items())*3])
        self.observation_space = spaces.Box(0, 1, shape=[len(self.joints.items())*3 + 2*3 + 4])
        self.__max_steps = nb_steps
        #p.addUserDebugText("Goal", self.robotTargetPos, [1,0,1])

    def step(self, action):
        self.episode_over = False
        self.current_steps += 1
        currentPos, currentOrientation = p.getBasePositionAndOrientation(self.robotId)
        currentOrientation = p.getEulerFromQuaternion(currentOrientation)
        if abs(currentOrientation[1]) > np.pi/3 or currentPos[2] < 0.3:
            self._reset_pos()
        self._set_action(action)
        if self.current_steps >= self.__max_steps:
            self.current_steps = 0
            self.episode_over = True
        return self._get_observation(), self._get_reward(), self.episode_over, {}

    def reset(self):
        self.current_steps = 0
        self._reset_pos()
        return self._get_observation()

    def render(self, mode='human', close=False):
        pass

    def dispose(self):
        p.disconnect()
        time.sleep(5)

    def _set_action(self, action):
        n_actuators = len(self.joints.items())
        assert action.shape == (n_actuators*3,)
        action = action.copy()
        action = np.clip(action, -1, 1)
        positions = []
        velocities = []
        forces = []
        for i in range(len(action)//3):
            #positions.append(action[i]*(self.joints[i][9]-self.joints[i][8]))
            positions.append((action[i*3]+1)/2*np.pi*2)
            velocities.append((action[i*3+1]+1)/2*self.joints[i][11])
            forces.append((action[i*3+2]+1)/2*self.joints[i][10])
        p.setJointMotorControlArray(self.robotId, range(n_actuators), p.POSITION_CONTROL, positions, velocities, forces)
        p.stepSimulation()

    def _get_observation(self):
        n_actuators = len(self.joints.items())
        observations = []
        states = p.getJointStates(self.robotId, range(n_actuators))
        for i, state in enumerate(states):
            #observations.append(state[0]/self.joints[i][9])
            observations.append(state[0]/2*np.pi)
            observations.append(state[1]/self.joints[i][11])
            observations.append(state[3]/self.joints[i][10])
        contactJoints = [joint for k, joint in self.joints.items() if b'lower_leg' in joint[1]]
        contacts = [-1,-1,-1,-1]
        for i, joint in enumerate(contactJoints):
            if len(p.getContactPoints(self.robotId, self.planeId, joint[0])) > 0:
                contacts[i] = 1
        currentRobotPos, currentRobotOrientation = p.getBasePositionAndOrientation(self.robotId)
        currentRobotOrientation = p.getEulerFromQuaternion(currentRobotOrientation)
        observations = (observations +
                        [x for x in p.getBaseVelocity(self.robotId)[0]] +
                        [x for x in p.getBaseVelocity(self.robotId)[1]] +
                        contacts)
        observations = np.asarray(observations)
        return observations

    def _get_reward(self):
        currentRobotPos, currentRobotOrientation = p.getBasePositionAndOrientation(self.robotId)
        #distance = np.linalg.norm(np.array(currentRobotPos)-np.array(self.robotTargetPos))
        #rotation = np.linalg.norm(np.array(p.getEulerFromQuaternion(p.getDifferenceQuaternion(currentRobotOrientation, self.robotTargetOrientation))))
        linearVelocity, angularVelocity = p.getBaseVelocity(self.robotId)
        #return 100/(distance*0.1+rotation)*linearVelocity[0]-abs(p.getEulerFromQuaternion(currentRobotOrientation)[1])/(2*np.pi)*100
        reward = np.linalg.norm(np.array(currentRobotPos[:2])-np.array(self.robotStartPos[:2]))
        currentRobotOrientation = p.getEulerFromQuaternion(currentRobotOrientation)
        if abs(currentRobotOrientation[1]) > np.pi/3 or currentRobotPos[2] < 0.3:
            reward -= 100
        if reward > 300:
            reward = 300
        return reward

    def _reset_pos(self):
        p.resetBasePositionAndOrientation(self.robotId, self.robotStartPos, self.robotStartOrientation)
        for i in range(len(self.joints.items())):
            p.resetJointState(self.robotId, self.joints[i][0], 0)
        p.removeAllUserDebugItems()
        randomOffsetPos = [
            random.random()*10,
            random.random()*10,
            0
        ]
        #self.robotTargetPos = [sum(x) for x in zip(self.robotStartPos, randomOffsetPos)]
        #self.robotTargetOrientation = p.getQuaternionFromEuler([0,0,random.random()])
        #p.addUserDebugText("Goal", self.robotTargetPos, [1,0,1])

