import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


""""Used for Pretraining"""

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True # Added
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)

        world.len_hist = 2

        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # go to the goal landmark itself.
        world.agents[0].goal_a = world.agents[0]
        world.agents[0].goal_b = np.random.choice(world.landmarks)

        world.agents[1].goal_a = world.agents[1]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])  #  I think the world has the default color.
        # random properties for landmarks
        world.landmarks[0].color = np.array([1, 0, 0])
        world.landmarks[1].color = np.array([0, 1, 0])
        world.landmarks[2].color = np.array([0, 0, 1])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        world.agents[1].goal_a.color = world.agents[1].goal_b.color                               
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2 #np.exp(-dist2)

    def observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos         
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks: # relative positions
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # the other agent's location


        location = [np.zeros(6)] # Empty info about the teammate
        # for other in world.agents:
        #     if other is agent: continue
        #     for entity in world.landmarks:
        #         location.append(entity.state.p_pos - other.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + location)

        # version 2, but change the benchmark_data to second one.
        # return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]])


    # Get the agent's velocity
    def benchmark_data(self, agent, world):
        return agent.state.p_vel

