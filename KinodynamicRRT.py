import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # [x, y, vx, vy]
        self.parent = parent
        self.path = [state]


class KinodynamicRRT:
    def __init__(
        self,
        start,
        goal,
        obstacles,
        bounds,
        max_iter=5000,
        dt=0.1,
        steer_time=0.5,
        goal_bias=0.2,
    ):
        self.start = Node(np.array(start, dtype=float))
        self.goal = Node(np.array(goal, dtype=float))
        self.obstacles = obstacles
        self.bounds = bounds.astype(float)
        self.max_iter = int(max_iter)
        self.dt = float(dt)
        self.steer_time = float(steer_time)
        self.goal_bias = float(goal_bias)
        self.nodes = [self.start]
        self.epsilon = 0.1
        self.kp = 1.0
        self.kv = 1.0

    def random_state(self):
        if np.random.random() < self.goal_bias:
            return self.goal.state
        x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
        vx = np.random.uniform(self.bounds[2, 0], self.bounds[2, 1])
        vy = np.random.uniform(self.bounds[3, 0], self.bounds[3, 1])
        return np.array([x, y, vx, vy], dtype=float)

    def nearest_node(self, state):
        pos_error = [np.linalg.norm(node.state[:2] - state[:2]) for node in self.nodes]
        vel_error = [np.linalg.norm(node.state[2:] - state[2:]) for node in self.nodes]
        distances = self.kp * np.array(pos_error) + self.kv * np.array(vel_error)
        return self.nodes[int(np.argmin(distances))]

    def steer(self, from_node, to_state):
        new_node = Node(np.copy(from_node.state), from_node)
        path = [new_node.state]

        for _ in range(int(self.steer_time / self.dt)):
            # 1. Calculate errors for position and velocity
            pos_error = to_state[:2] - new_node.state[:2]
            vel_error = to_state[2:] - new_node.state[2:]

            desired_acc = self.kp * pos_error + self.kv * vel_error

            # 3. Limit acceleration to stay within bounds
            limited_acc = np.clip(desired_acc, -1.0, 1.0)

            # 4. Update velocity and position
            new_vel = new_node.state[2:] + limited_acc * self.dt

            # 5. Limit velocity magnitude to 1
            vel_magnitude = np.linalg.norm(new_vel)
            if vel_magnitude > 1:
                new_vel = new_vel / vel_magnitude  # Normalize and scale

            new_pos = new_node.state[:2] + new_vel * self.dt

            # 5. Combine into new_state and check for validity
            new_state = np.concatenate([new_pos, new_vel])

            if self.is_valid(new_state):
                new_node.state = new_state
                path.append(new_state)
            else:
                break

        new_node.path = np.array(path)
        return new_node

    def is_valid(self, state):
        x, y = state[:2]
        if not (
            self.bounds[0, 0] <= x <= self.bounds[0, 1]
            and self.bounds[1, 0] <= y <= self.bounds[1, 1]
        ):
            return False
        for ox, oy, radius in self.obstacles:
            if np.hypot(x - ox, y - oy) <= radius + 0.1:
                return False
        return True

    def plan_with_visualization(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_plot(ax)

        (tree_lines,) = ax.plot([], [], "go", markersize=2, alpha=0.5)
        plt.ion()
        plt.show()

        for i in range(self.max_iter):
            rnd_state = self.random_state()
            nearest_node = self.nearest_node(rnd_state)
            new_node = self.steer(nearest_node, rnd_state)

            if new_node and self.is_valid(new_node.state):
                self.nodes.append(new_node)
                self.update_plot(ax, tree_lines, new_node)

                if (
                    np.linalg.norm(new_node.state[:2] - self.goal.state[:2])
                    < self.epsilon
                    and np.linalg.norm(new_node.state[2:] - self.goal.state[2:])
                    < self.epsilon
                ):
                    print(f"Goal reached after {i + 1} iterations!")
                    self.visualize_final_path(ax, new_node)
                    plt.ioff()
                    plt.show()
                    return self.get_path(new_node)

            if i % 100 == 0:
                plt.pause(0.001)

        print(f"Failed to reach the goal after {self.max_iter} iterations")
        plt.ioff()
        plt.show()
        return None

    def get_path(self, node):
        path = []
        while node:
            path = list(node.path) + path
            node = node.parent
        return np.array(path, dtype=float)

    def setup_plot(self, ax):
        for ox, oy, radius in self.obstacles:
            circle = Circle((ox, oy), radius, color="red", alpha=0.5)
            ax.add_artist(circle)

        ax.plot(
            self.start.state[0], self.start.state[1], "go", markersize=10, label="Start"
        )
        ax.plot(
            self.goal.state[0], self.goal.state[1], "ro", markersize=10, label="Goal"
        )

        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True)
        plt.title("Omnidirectional Kinodynamic RRT")

    def update_plot(self, ax, tree_lines, new_node):
        path = np.array(new_node.path)
        ax.plot(path[:, 0], path[:, 1], "g-", linewidth=0.5, alpha=0.5)

        x_coords = [node.state[0] for node in self.nodes]
        y_coords = [node.state[1] for node in self.nodes]
        tree_lines.set_data(x_coords, y_coords)

    def visualize_final_path(self, ax, goal_node):
        path = self.get_path(goal_node)
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="Final Path")
        ax.legend()
        plt.draw()

    def animate_path(self, path):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_plot(ax)

        # Plot the entire path
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, alpha=0.5, label="Path")

        # Create a point representing the agent
        (agent,) = ax.plot([], [], "go", markersize=10, label="Agent")

        # Create an arrow to represent velocity
        velocity_arrow = ax.arrow(0, 0, 0, 0, color="r", width=0.05, head_width=0.2)

        # Create a text object to display velocity
        velocity_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            agent.set_data([], [])
            velocity_arrow.set_visible(False)
            velocity_text.set_text("")
            return agent, velocity_arrow, velocity_text

        def animate(i):
            nonlocal velocity_arrow
            state = path[i]
            agent.set_data([state[0]], [state[1]])

            # Remove the old arrow and create a new one
            velocity_arrow.remove()
            velocity_arrow = ax.arrow(
                state[0],
                state[1],
                state[2] * 0.5,
                state[3] * 0.5,
                color="r",
                width=0.05,
                head_width=0.2,
            )

            # Update velocity text
            velocity = np.linalg.norm(state[2:])
            velocity_text.set_text(f"Velocity: {velocity:.2f}")

            return agent, velocity_arrow, velocity_text

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(path),
            interval=50,
            blit=True,
            repeat=False,
        )

        # anim.save('kinodynamic_rrt_animation.gif', writer='pillow', fps=10)

        plt.legend()
        plt.show()

        return anim


def main():
    start = [0.0, 0.0, 0.0, 0.0]  # [x, y, vx, vy]
    goal = [10.0, 10.0, 0.0, 0.0]
    obstacles = [(2.0, 2.0, 1.0), (5.0, 5.0, 1.5), (8.0, 8.0, 1.5)]  # [x, y, radius]
    bounds = np.array(
        [[-2.0, 12.0], [-2.0, 12.0], [-1.0, 1.0], [-1.0, 1.0]]
    )  # [x_min, x_max], [y_min, y_max], [vx_min, vx_max], [vy_min, vy_max]

    rrt = KinodynamicRRT(
        start,
        goal,
        obstacles,
        bounds,
        max_iter=5000,
        dt=0.1,
        steer_time=1.0,
        goal_bias=0.1,
    )
    path = rrt.plan_with_visualization()

    if path is not None:
        print("Path found!")
        rrt.animate_path(path)
    else:
        print("No path found")


if __name__ == "__main__":
    main()
