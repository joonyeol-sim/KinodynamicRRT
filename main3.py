import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
rewire_radius = 2.0
goal_threshold = 1.0
x_range = (0, 10)
y_range = (0, 10)
theta_range = (0, 2 * np.pi)
max_iterations = 1000
dt = 0.1
num_steps = 10

class Node:
    def __init__(self, state, parent=None, cost=0, trajectory=None):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.trajectory = trajectory if trajectory is not None else []

    def __repr__(self):
        return f"Node(state={self.state}, cost={self.cost})"

class DifferentialDriveRobot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta])

    def state_transition(self, state, control, dt):
        x, y, theta = state
        v, omega = control

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt

        return np.array([x_new, y_new, theta_new])

    def get_state(self):
        return self.state

# Sample State
def sample_state(x_range, y_range, theta_range):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    theta = random.uniform(*theta_range)
    return np.array([x, y, theta])

def distance(state1, state2):
    return np.linalg.norm(state1[:2] - state2[:2])

def nearest_neighbor(tree, sample):
    return min(tree, key=lambda node: distance(node.state, sample))

def is_collision_free(trajectory):
    # Simple collision checking logic (assuming no obstacles)
    return True

def steer(robot, from_node, to_state, dt, num_steps):
    trajectory = []
    state = from_node.state
    for _ in range(num_steps):
        dx = to_state[0] - state[0]
        dy = to_state[1] - state[1]
        theta = np.arctan2(dy, dx)
        v = min(1.0, distance(state, to_state))
        omega = min(1.0, theta - state[2])
        control = [v, omega]
        state = robot.state_transition(state, control, dt)
        trajectory.append(state)
        if distance(state, to_state) < dt:
            break
    return trajectory if is_collision_free(trajectory) else None

def kinodynamic_rrt_star(start_state, goal_state, x_range, y_range, theta_range, max_iterations, dt, num_steps):
    robot = DifferentialDriveRobot(*start_state)
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    tree = [start_node]

    # Plot initialization
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.plot(start_state[0], start_state[1], 'go', label='Start')
    ax.plot(goal_state[0], goal_state[1], 'ro', label='Goal')
    ax.legend()
    plt.title("Kinodynamic RRT* Path")

    for _ in range(max_iterations):
        sample = sample_state(x_range, y_range, theta_range)
        nearest = nearest_neighbor(tree, sample)
        trajectory = steer(robot, nearest, sample, dt, num_steps)

        if trajectory:
            new_state = trajectory[-1]
            new_cost = nearest.cost + len(trajectory) * dt
            new_node = Node(new_state, parent=nearest, cost=new_cost, trajectory=trajectory)

            # Rewire the tree
            for node in tree:
                if distance(node.state, new_state) < rewire_radius and new_cost + distance(new_state, node.state) < node.cost:
                    node.parent = new_node
                    node.cost = new_cost + distance(new_state, node.state)

            tree.append(new_node)

            # Plot new node and trajectory
            ax.plot([state[0] for state in trajectory], [state[1] for state in trajectory], '-b')
            ax.plot(new_state[0], new_state[1], 'bo')
            plt.draw()
            plt.pause(0.01)  # Pause to update the plot

            if distance(new_state, goal_state) < goal_threshold:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + distance(new_state, goal_state)
                path = construct_path(goal_node)
                plot_path(path, ax)
                plt.ioff()
                plt.show()
                return path
    plt.ioff()
    plt.show()
    return None

def construct_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append((node.state, node.trajectory))
        node = node.parent
    path.reverse()
    return path

def plot_path(path, ax):
    if path:
        for state, trajectory in path:
            x = [state[0]] + [s[0] for s in trajectory]
            y = [state[1]] + [s[1] for s in trajectory]
            ax.plot(x, y, '-o')
    else:
        print("No path found")

def main():
    start_state = [0, 0, 0]
    goal_state = [9, 9, 0]

    path = kinodynamic_rrt_star(start_state, goal_state, x_range, y_range, theta_range, max_iterations, dt, num_steps)
    print(path)

if __name__ == "__main__":
    main()
