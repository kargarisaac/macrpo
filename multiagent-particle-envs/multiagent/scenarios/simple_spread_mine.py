import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

N_AGENTS = 3
WORLD_SIZE = 1


class Scenario(BaseScenario):
    def make_world(self):
        # np.random.seed(np.random.randint(1, 1000))
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N_AGENTS
        num_landmarks = N_AGENTS
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.07
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        colors = [
            np.array([0.35, 0.35, 0.85]),
            np.array([0.5, 0.6, 0.1]),
            np.array([0.8, 0.1, 0.3]),
            np.array([0.8, 0.1, 0.5]),
            np.array([0.8, 0.4, 0.3]),
            np.array([0.4, 0.1, 0.3]),
            np.array([0.6, 0.1, 0.3]),
            np.array([0.8, 0.2, 0.5]),
            np.array([0.3, 0.1, 0.8]),
            np.array([0.4, 0.87, 0.4])
        ]
        for i, agent in enumerate(world.agents):
            agent.color = colors[i]
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = colors[i] + np.array([0.1, 0.1, 0.1])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(
                -WORLD_SIZE, +WORLD_SIZE, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(
                -WORLD_SIZE, +WORLD_SIZE, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        agent_id = int(agent.name.split()[1])
        l = world.landmarks[agent_id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        rew -= dist

        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    
    def reward(self, agent, world):
        # Agents are rewarded based on distance to its own landmark, penalized for collisions
        rew = 0
        agent_id = int(agent.name.split()[1])
        l = world.landmarks[agent_id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        rew -= dist

        # TODO: test if reaching landmark needs positive reward
        # if dist < 0.1: rew += 1

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def done(self, agent, world):
        # go out of boarder
        x = abs(agent.state.p_pos[0])
        y = abs(agent.state.p_pos[1])
        if (x > 1.2 or y > 1.2):
            return True

        # reach landmark
        agent_id = int(agent.name.split()[1])
        l = world.landmarks[agent_id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        if dist < 0.1:
            return True

        return False
