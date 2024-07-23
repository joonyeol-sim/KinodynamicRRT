import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random
from scipy.optimize import minimize

np.set_printoptions(suppress=True, precision=8, floatmode="fixed")


@dataclass
class Node:
    state: np.ndarray
    cost: float = float("inf")
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
        return np.array_equal(self.state, other.state) and self.cost == other.cost

    def __hash__(self):
        return hash(tuple(self.state) + (self.cost,))


class KinodynamicRRTStar:
    def __init__(
        self,
        start: List[float],
        goal: List[float],
        obstacles: List[Tuple[float, float, float]],
        bounds: np.ndarray,
        max_iter: int = 1500,
        dt: float = 0.1,
        goal_bias: float = 0.15,
        connect_circle_dist: float = float("inf"),
        seed: Optional[int] = None,
    ):
        self.start = Node(state=np.array(start, dtype=float))
        self.goal = Node(state=np.array(goal, dtype=float))
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_iter = max_iter
        self.dt = dt
        self.goal_bias = float(goal_bias)
        self.connect_circle_dist = connect_circle_dist
        self.start.cost = 0.0
        self.nodes = [self.start]

        self.max_acceleration_x = 1.0
        self.max_acceleration_y = 1.0

        self.epsilon = 0.1
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def calculate_t1_t2(x0, v0, xf, vf, a_max):
        a = a_max
        b = 2 * v0
        c = (v0**2 - vf**2) / (2 * a_max) + x0 - xf

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None, None

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (v0 - vf) / a + t1

        return t1, t2

    def steer(self, from_node, to_state, callback=None) -> Optional[Node]:
        new_node = Node(np.copy(from_node.state), from_node)
        path = [new_node.state]
        total_time = 0.0

        max_acceleration_x = self.max_acceleration_x
        max_acceleration_y = self.max_acceleration_y

        # x와 y 방향에 대해 각각 t1, t2 계산
        t1_x, t2_x = self.calculate_t1_t2(
            new_node.state[0],
            new_node.state[2],
            to_state[0],
            to_state[2],
            max_acceleration_x,
        )
        t1_y, t2_y = self.calculate_t1_t2(
            new_node.state[1],
            new_node.state[3],
            to_state[1],
            to_state[3],
            max_acceleration_y,
        )

        # t1과 t2가 계산되지 않으면 (해가 없으면) None 반환
        if t1_x is None or t1_y is None:
            return None

        # 전체 제어 시간 계산
        total_time_x = t1_x + t2_x
        total_time_y = t1_y + t2_y

        total_control_time = max(total_time_x, total_time_y)

        x0 = new_node.state[0]
        v0_x = new_node.state[2]

        y0 = new_node.state[1]
        v0_y = new_node.state[3]

        v1_x = v0_x + max_acceleration_x * t1_x
        x1 = x0 + v0_x * t1_x + 0.5 * max_acceleration_x * t1_x**2
        v2_x = v1_x - max_acceleration_x * t2_x
        x2 = x1 + v1_x * t2_x - 0.5 * max_acceleration_x * t2_x**2

        v1_y = v0_y + max_acceleration_y * t1_y
        y1 = y0 + v0_y * t1_y + 0.5 * max_acceleration_y * t1_y**2
        v2_y = v1_y - max_acceleration_y * t2_y
        y2 = y1 + v1_y * t2_y - 0.5 * max_acceleration_y * t2_y**2

        assert (x2 - to_state[0]) < 1e-6
        assert (y2 - to_state[1]) < 1e-6
        assert (v2_x - to_state[2]) < 1e-6
        assert (v2_y - to_state[3]) < 1e-6

        while total_time < total_control_time:
            acc = np.zeros(2)

            # x 방향 제어
            if total_time < t1_x:
                acc[0] = max_acceleration_x
            elif total_time < t1_x + t2_x:
                acc[0] = -max_acceleration_x
            elif total_time >= total_time_x:
                if total_time_y > total_time_x:
                    if total_time < total_time_x + (total_time_y - total_time_x) / 2:
                        acc[0] = -4 * to_state[2] / (total_time_y - total_time_x)
                    elif total_time_x + (total_time_y - total_time_x) / 2 < total_time:
                        acc[0] = 4 * to_state[2] / (total_time_y - total_time_x)
                    print(acc[0])

            # y 방향 제어
            if total_time < t1_y:
                acc[1] = max_acceleration_y
            elif total_time < t1_y + t2_y:
                acc[1] = -max_acceleration_y
            elif total_time >= total_time_y:
                if total_time_x > total_time_y:
                    if total_time < total_time_y + (total_time_x - total_time_y) / 2:
                        acc[1] = -4 * to_state[3] / (total_time_x - total_time_y)
                    elif total_time_y + (total_time_x - total_time_y) / 2 < total_time:
                        acc[1] = 4 * to_state[3] / (total_time_x - total_time_y)
                    print(acc[1])

            # 속도 업데이트
            new_vel = new_node.state[2:] + acc * self.dt

            # 속도 제한
            # new_vel = np.clip(new_vel, self.bounds[2, 0], self.bounds[2, 1])

            # 위치 업데이트
            new_pos = (
                new_node.state[:2]
                + new_node.state[2:] * self.dt
                + 0.5 * acc * self.dt**2
            )

            # 새로운 상태 생성
            new_state = np.concatenate([new_pos, new_vel])
            print(new_state)

            if self.is_valid(new_state):
                new_node.state = new_state
                path.append(new_state)
                if callback:
                    callback(path, to_state)
            else:
                break

            total_time += self.dt

        # 최종 상태 추가
        if self.is_valid(to_state):
            # new_node.state = to_state
            # path.append(to_state)
            # if callback:
            #     callback(path, to_state)

            new_node.path = np.array(path)
            new_node.cost = from_node.cost + self.calculate_distance(
                new_node.state, from_node.state
            )
            return new_node
        return None

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

    def get_random_state(self) -> np.ndarray:
        if np.random.random() < self.goal_bias:
            return self.goal.state
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
            y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
            vx = np.random.uniform(self.bounds[2, 0], self.bounds[2, 1])
            vy = np.random.uniform(self.bounds[3, 0], self.bounds[3, 1])
            state = np.array([x, y, vx, vy], dtype=float)
            if self.is_valid(state):
                return state
        print("Warning: Failed to find a valid random state after maximum attempts")
        return None

    def calculate_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        pos_error = np.linalg.norm(state1[:2] - state2[:2])
        vel_error = np.linalg.norm(state1[2:] - state2[2:])
        return pos_error + vel_error

    def get_nearest_node(self, state: np.ndarray) -> Node:
        distances = [self.calculate_distance(node.state, state) for node in self.nodes]
        return self.nodes[int(np.argmin(distances))]

    def get_near_nodes(self, state: np.ndarray) -> List[Node]:
        return [
            node
            for node in self.nodes
            if self.calculate_distance(node.state, state) <= self.connect_circle_dist
        ]

    def is_near_goal(self, node: Node) -> bool:
        pos_diff = np.linalg.norm(node.state[:2] - self.goal.state[:2])
        vel_diff = np.linalg.norm(node.state[2:] - self.goal.state[2:])
        return pos_diff < self.epsilon and vel_diff < self.epsilon

    def choose_parent(self, node: Node, near_nodes: List[Node]) -> Node:
        min_cost = float("inf")
        best_parent = None
        for near_node in near_nodes:
            new_node = self.steer(near_node, node.state)
            if new_node and self.is_valid(new_node.state):
                cost = new_node.cost
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_node
                    node = new_node
        if best_parent:
            node.parent = best_parent
            node.cost = min_cost
        return node

    def rewire(self, node: Node, near_nodes: List[Node]):
        for near_node in near_nodes:
            if near_node != node.parent:
                new_node = self.steer(node, near_node.state)
                if new_node and self.is_valid(new_node.state):
                    new_cost = node.cost + self.calculate_distance(
                        node.state, new_node.state
                    )
                    if new_cost < near_node.cost:
                        near_node.parent = node
                        near_node.cost = new_cost
                        near_node.path = new_node.path

    def plan_with_visualization(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_plot(ax)
        (tree_lines,) = ax.plot([], [], "go", markersize=2, alpha=0.5)
        (best_path_line,) = ax.plot([], [], "r-", linewidth=2, label="Best Path")
        (steer_path_line,) = ax.plot(
            [], [], "y-", linewidth=1, alpha=0.5, label="Steer Path"
        )
        (to_state_point,) = ax.plot([], [], "b*", markersize=10, label="To State")
        plt.ion()
        plt.show()

        def update_steer_path(path, to_state):
            path = np.array(path)
            steer_path_line.set_data(path[:, 0], path[:, 1])
            to_state_point.set_data([to_state[0]], [to_state[1]])
            plt.pause(0.01)

        best_goal_node = None
        for i in range(self.max_iter):
            print(f"Iteration {i + 1}/{self.max_iter}")
            rnd_state = self.get_random_state()
            nearest_node = self.get_nearest_node(rnd_state)
            # new_node = self.steer(nearest_node, rnd_state, update_steer_path)
            new_node = self.steer(nearest_node, rnd_state)
            if new_node and self.is_valid(new_node.state):
                new_node.parent = nearest_node
                self.nodes.append(new_node)
                self.update_plot(ax, tree_lines, new_node)
                print(f"New node added at {new_node.state}")
                if self.is_near_goal(new_node):
                    print("Goal reached!")
                    best_goal_node = new_node
                    best_cost = new_node.cost
                    break
            if i % 1000 == 0:
                plt.pause(0.001)

        if best_goal_node:
            self.visualize_final_path(ax, best_goal_node)
            plt.ioff()
            plt.show()
            return self.get_path(best_goal_node)
        else:
            print(f"Failed to reach the goal after {self.max_iter} iterations")
            plt.ioff()
            plt.show()
            return None

    def update_best_path_plot(self, ax, best_path_line, goal_node):
        path = self.get_path(goal_node)
        best_path_line.set_data(path[:, 0], path[:, 1])
        ax.legend()
        plt.draw()

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
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, alpha=0.5, label="Path")
        (agent,) = ax.plot([], [], "go", markersize=10, label="Agent")
        velocity_arrow = ax.arrow(0, 0, 0, 0, color="r", width=0.05, head_width=0.2)
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
        plt.legend()
        plt.show()
        return anim


def main():
    start = [0.0, 0.0, 0.0, 0.0]  # [x, y, vx, vy]
    goal = [10.0, 10.0, 0.0, 0.0]
    # obstacles = [(2.0, 2.0, 1.0), (5.0, 5.0, 1.5), (8.0, 8.0, 1.5)]  # [x, y, radius]
    obstacles = []  # [x, y, radius]
    bounds = np.array(
        [[-2.0, 12.0], [-2.0, 12.0], [-1.0, 1.0], [-1.0, 1.0]]
    )  # [x_min, x_max], [y_min, y_max], [vx_min, vx_max], [vy_min, vy_max]
    rrt = KinodynamicRRTStar(
        start, goal, obstacles, bounds, max_iter=1500, dt=0.1, goal_bias=0.1, seed=42
    )
    path = rrt.plan_with_visualization()
    if path is not None:
        print("Path found!")
        rrt.animate_path(path)
    else:
        print("No path found")


if __name__ == "__main__":
    main()
