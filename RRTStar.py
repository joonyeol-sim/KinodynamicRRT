import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random

np.set_printoptions(suppress=True, precision=8, floatmode="fixed")


def bang_bang_transform(path):
    controls = []

    for i in range(len(path) - 1):
        q = path[i]
        q_next = path[i + 1]

        v = q_next - q
        v_norm = np.linalg.norm(v)
        v_hat = v / v_norm if v_norm > 0 else v

        s = np.max(np.abs(v_hat))
        a = v_hat / s if s > 0 else 0
        t = np.sqrt(s * v_norm)

        # Bang-bang control: accelerate, then decelerate
        control1 = (a, t)
        control2 = (-a, t)

        controls.extend([control1, control2])

        q_1 = q + 0.5 * a * t ** 2
        v_1 = a * t
        q_f = q_1 + v_1 * t - 0.5 * a * t ** 2
        v_f = v_1 - a * t
        assert np.allclose(q_f, q_next), f"{q_f} != {q_next}"
        assert np.allclose(v_f, 0), f"{v_f} != {0}"

    return controls

@dataclass
class Node:
    state: np.ndarray
    parent: Optional["Node"] = None
    path: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.state, np.ndarray):
            self.state = np.array(self.state, dtype=float)
        if not self.path:
            self.path = [self.state]

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(tuple(self.state))

    def __repr__(self):
        return f"Node(state={self.state}, parent={self.parent})"

class RRT:
    def __init__(
        self,
        start: List[float],
        goal: List[float],
        obstacles: List[Tuple[float, float, float]],
        bounds: np.ndarray,
        max_iter: int = 1000,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.start = Node(state=np.array(start, dtype=float))
        self.goal = Node(state=np.array(goal, dtype=float))
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.nodes = [self.start]

        self.epsilon = 0.1
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.tree_lines, = self.ax.plot([], [], "go", markersize=2, alpha=0.5)
        self.path_line, = self.ax.plot([], [], "r-", linewidth=2, label="Path")

    def steer(self, from_node: Node, to_state: np.ndarray) -> Node:
        dir_vector = to_state - from_node.state
        dist = np.linalg.norm(dir_vector)
        if dist > self.step_size:
            dir_vector = dir_vector / dist * self.step_size
        new_state = from_node.state + dir_vector
        new_node = Node(new_state, parent=from_node)
        new_node.path = from_node.path + [new_state]
        return new_node

    def is_valid(self, state: np.ndarray) -> bool:
        x, y = state[:2]
        if not (self.bounds[0, 0] <= x <= self.bounds[0, 1] and
                self.bounds[1, 0] <= y <= self.bounds[1, 1]):
            return False
        for ox, oy, radius in self.obstacles:
            if np.hypot(x - ox, y - oy) <= radius + 0.1:
                return False
        return True

    def get_random_state(self) -> np.ndarray:
        if np.random.random() < self.goal_sample_rate:
            return self.goal.state
        return np.array([
            np.random.uniform(self.bounds[0, 0], self.bounds[0, 1]),
            np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
        ])

    def get_nearest_node(self, state: np.ndarray) -> Node:
        distances = [np.linalg.norm(node.state - state) for node in self.nodes]
        return self.nodes[int(np.argmin(distances))]

    def is_near_goal(self, node: Node) -> bool:
        return np.linalg.norm(node.state - self.goal.state) < self.epsilon

    def plan(self):
        self.setup_plot(self.ax)
        plt.ion()
        plt.show()

        for i in range(self.max_iter):
            rnd_state = self.get_random_state()
            nearest_node = self.get_nearest_node(rnd_state)
            new_node = self.steer(nearest_node, rnd_state)

            if self.is_valid(new_node.state):
                self.nodes.append(new_node)
                self.update_plot(self.ax, self.tree_lines)

                if self.is_near_goal(new_node):
                    self.goal.parent = new_node
                    self.goal.path = new_node.path + [self.goal.state]
                    self.visualize_path(self.ax, self.goal)
                    plt.ioff()
                    plt.show()
                    return self.get_path(self.goal)

            if i % 10 == 0:
                plt.pause(0.001)

        print(f"Failed to reach the goal after {self.max_iter} iterations")
        plt.ioff()
        plt.show()
        return None

    def get_path(self, node: Node):
        return np.array(node.path)

    def plan_with_bang_bang(self):
        path = self.plan()  # Get the basic path from RRT
        if path is not None:
            controls = bang_bang_transform(path)
            return path, controls
        return None, None

    def setup_plot(self, ax):
        for ox, oy, radius in self.obstacles:
            circle = Circle((ox, oy), radius, color="red", alpha=0.5)
            ax.add_artist(circle)
        ax.plot(self.start.state[0], self.start.state[1], "go", markersize=10, label="Start")
        ax.plot(self.goal.state[0], self.goal.state[1], "ro", markersize=10, label="Goal")
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True)
        plt.title("RRT")

    def update_plot(self, ax, tree_lines):
        for line in ax.lines:
            if line not in [self.path_line, tree_lines]:
                line.remove()

        for node in self.nodes:
            if node.parent:
                ax.plot([node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]],
                        "g-", linewidth=0.5, alpha=0.5)

        x_coords = [node.state[0] for node in self.nodes]
        y_coords = [node.state[1] for node in self.nodes]
        tree_lines.set_data(x_coords, y_coords)

        plt.draw()

    def visualize_path(self, ax, goal_node):
        path = self.get_path(goal_node)
        self.path_line.set_data(path[:, 0], path[:, 1])
        plt.draw()

    def animate_path(self, path):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_plot(ax)
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, alpha=0.5, label="Path")
        agent, = ax.plot([], [], "go", markersize=10, label="Agent")

        def init():
            agent.set_data([], [])
            return agent,

        def animate(i):
            state = path[i]
            agent.set_data([state[0]], [state[1]])
            return agent,

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(path),
            interval=50,
            blit=True,
            repeat=False,
        )
        plt.legend()
        plt.show()
        return anim

    def animate_bang_bang_path(self, path, controls):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_plot(ax)

        # 전체 경로 그리기
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], "b-", linewidth=2, alpha=0.5, label="Path")

        agent, = ax.plot([], [], "go", markersize=10, label="Agent")
        velocity_arrow = ax.arrow(0, 0, 0, 0, color="r", width=0.05, head_width=0.2)
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        total_time = sum(t for _, t in controls)

        def init():
            agent.set_data([], [])
            velocity_arrow.set_visible(False)
            time_text.set_text("")
            return agent, velocity_arrow, time_text

        def animate(frame):
            nonlocal velocity_arrow
            current_time = frame * total_time / 500  # 500 프레임으로 가정

            # 현재 시간에 해당하는 상태 찾기
            cumulative_time = 0
            pos = path[0]
            vel = np.zeros_like(pos)

            for i, (a, t) in enumerate(controls):
                if cumulative_time + t > current_time:
                    delta_t = current_time - cumulative_time
                    pos = pos + vel * delta_t + 0.5 * a * delta_t ** 2
                    vel = vel + a * delta_t
                    break
                else:
                    pos = pos + vel * t + 0.5 * a * t ** 2
                    vel = vel + a * t
                cumulative_time += t

            agent.set_data([pos[0]], [pos[1]])

            # 속도 화살표 업데이트
            velocity_arrow.remove()
            velocity_arrow = ax.arrow(pos[0], pos[1], vel[0] * 0.5, vel[1] * 0.5,
                                      color="r", width=0.05, head_width=0.2)

            time_text.set_text(f"Time: {current_time:.2f} / {total_time:.2f}")

            return agent, velocity_arrow, time_text

        anim = FuncAnimation(fig, animate, init_func=init, frames=500,
                             interval=20, blit=True, repeat=False)
        plt.legend()
        plt.show()
        return anim

def main():
    start = [0.0, 0.0]
    goal = [10.0, 10.0]
    obstacles = [(2.0, 2.0, 1.0), (5.0, 5.0, 1.5), (8.0, 8.0, 1.5)]
    bounds = np.array([[-2.0, 12.0], [-2.0, 12.0]])
    rrt = RRT(start, goal, obstacles, bounds, max_iter=5000, step_size=1.5, goal_sample_rate=0.1)
    path, controls = rrt.plan_with_bang_bang()
    if path is not None:
        print("Path found!")
        rrt.animate_bang_bang_path(path, controls)
    else:
        print("No path found")

if __name__ == "__main__":
    main()