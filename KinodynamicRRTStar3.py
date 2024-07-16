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
        # Ensure state is a numpy array
        if not isinstance(self.state, np.ndarray):
            self.state = np.array(self.state, dtype=float)

        # Initialize path if empty
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

        # 위치에 대한 PID 게인
        self.kp_pos = 1.0
        self.ki_pos = 0.1
        self.kd_pos = 0.5

        # 속도에 대한 PID 게인
        self.kp_vel = 1.0
        self.ki_vel = 0.1
        self.kd_vel = 0.5

        # 적분 오차 초기화
        self.integral_error_pos = np.zeros(2)
        self.integral_error_vel = np.zeros(2)

        # 적분항 최대값 (anti-windup)
        self.max_integral_pos = 10.0
        self.max_integral_vel = 5.0

        self.max_steering_time = float("inf")
        self.epsilon = 0.2

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_random_state(self) -> np.ndarray:
        if np.random.random() < self.goal_bias:
            return self.goal.state

        max_attempts = 100  # 최대 시도 횟수 설정
        for _ in range(max_attempts):
            x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
            y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
            vx = np.random.uniform(self.bounds[2, 0], self.bounds[2, 1])
            vy = np.random.uniform(self.bounds[3, 0], self.bounds[3, 1])
            state = np.array([x, y, vx, vy], dtype=float)

            if self.is_valid(state):
                return state

        # 유효한 상태를 찾지 못한 경우, 경고를 출력하고 None을 반환
        print("Warning: Failed to find a valid random state after maximum attempts")
        return None

    def calculate_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        pos_error = np.linalg.norm(state1[:2] - state2[:2])
        vel_error = np.linalg.norm(state1[2:] - state2[2:])
        return self.kp_pos * pos_error + self.kp_vel * vel_error

    def get_nearest_node(self, state: np.ndarray) -> Node:
        distances = [self.calculate_distance(node.state, state) for node in self.nodes]
        return self.nodes[int(np.argmin(distances))]

    def get_near_nodes(self, state: np.ndarray) -> List[Node]:
        return [
            node
            for node in self.nodes
            if self.calculate_distance(node.state, state) <= self.connect_circle_dist
        ]

    def steer(self, from_node, to_state, callback=None) -> Optional[Node]:
        new_node = Node(np.copy(from_node.state), from_node)
        path = [new_node.state]
        total_time = 0.0

        # 적분 오차 및 이전 오차 초기화
        self.integral_error_pos = np.zeros(2)
        self.integral_error_vel = np.zeros(2)
        prev_error_pos = np.zeros(2)
        prev_error_vel = np.zeros(2)

        while total_time < self.max_steering_time:
            # 1. Calculate errors for position and velocity
            error_pos = to_state[:2] - new_node.state[:2]
            error_vel = to_state[2:] - new_node.state[2:]
            print(f"Position error: {error_pos}")
            print(f"Velocity error: {error_vel}")

            error = error_pos - 0.5 * self.dt * (to_state[2:] + new_node.state[2:])

            # 적분 오차 업데이트
            self.integral_error_pos += error_pos * self.dt
            self.integral_error_vel += error_vel * self.dt

            # Anti-windup: 적분항 제한
            self.integral_error_pos = np.clip(
                self.integral_error_pos, -self.max_integral_pos, self.max_integral_pos
            )
            self.integral_error_vel = np.clip(
                self.integral_error_vel, -self.max_integral_vel, self.max_integral_vel
            )

            # 미분항 계산
            derivative_pos = (error_pos - prev_error_pos) / self.dt
            derivative_vel = (error_vel - prev_error_vel) / self.dt

            # 2. Calculate acceleration (PID 제어)
            acc_pos = (
                self.kp_pos * error_pos
                + self.ki_pos * self.integral_error_pos
                + self.kd_pos * derivative_pos
            )

            acc_vel = (
                self.kp_vel * error_vel
                + self.ki_vel * self.integral_error_vel
                + self.kd_vel * derivative_vel
            )

            # acc_pos = (
            #     2
            #     * (new_node.state[:2] - to_state[:2] - to_state[2:] * self.dt)
            #     / (self.dt**2)
            # )
            # acc_vel = (new_node.state[2:] - to_state[2:]) / self.dt
            #
            # w_pos = 0.5
            # w_vel = 0.5

            # 최종 가속도 계산
            # acc = w_pos * acc_pos + w_vel * acc_vel
            acc = acc_pos + acc_vel
            if np.linalg.norm(acc) < 0.001:
                break
            print(f"Acceleration: {acc}")

            # 3. Limit acceleration to stay within bounds
            limited_acc = np.clip(acc, -1.0, 1.0)

            # 4. Update velocity
            new_vel = new_node.state[2:] + limited_acc * self.dt

            # 5. Limit velocity to stay within bounds
            new_vel = np.clip(new_vel, self.bounds[2, 0], self.bounds[2, 1])
            print(f"New velocity: {new_vel}")

            # 6. Update position
            new_pos = (
                new_node.state[:2]
                + new_node.state[2:] * self.dt
                + 0.5 * limited_acc * self.dt**2
            )
            print(f"New position: {new_pos}")

            # 7. Combine into new_state and check for validity
            new_state = np.concatenate([new_pos, new_vel])
            print(f"New state: {new_state}")

            if self.is_valid(new_state):
                new_node.state = new_state
                path.append(new_state)
                if callback:
                    callback(path, to_state)
            else:
                break

            # 8. Check if the new state is close enough to the goal
            print(f"Current state: {new_state}")
            print(f"to_state: {to_state}")
            print(
                f"Distance to to state: {self.calculate_distance(new_state, to_state)}"
            )
            if self.calculate_distance(new_state, to_state) < self.epsilon:
                new_node.path = np.array(path)
                new_node.cost = from_node.cost + self.calculate_distance(
                    new_node.state, from_node.state
                )
                return new_node if len(path) > 1 else None

            # 이전 오차 업데이트
            prev_error_pos = error_pos
            prev_error_vel = error_vel

            total_time += self.dt

        # new_node.path = np.array(path)
        # new_node.cost = from_node.cost + self.calculate_distance(
        #     new_node.state, from_node.state
        # )
        # return new_node if len(path) > 1 else None
        return None

    def is_near_goal(self, node: Node) -> bool:
        pos_diff = np.linalg.norm(node.state[:2] - self.goal.state[:2])
        vel_diff = np.linalg.norm(node.state[2:] - self.goal.state[2:])
        return pos_diff < self.epsilon and vel_diff < self.epsilon

    def is_valid(self, state: np.ndarray) -> bool:
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

    def choose_parent(self, node: Node, near_nodes: List[Node]) -> Node:
        min_cost = float("inf")
        best_parent = None
        for near_node in near_nodes:
            new_node = self.steer(near_node, node.state)
            if new_node and self.is_collision_free(near_node, new_node):
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
                if new_node and self.is_collision_free(node, new_node):
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
            new_node = self.steer(nearest_node, rnd_state)
            # new_node = self.steer(nearest_node, rnd_state, update_steer_path)

            if new_node and self.is_valid(new_node.state):
                new_node.parent = nearest_node
                self.nodes.append(new_node)
                self.update_plot(ax, tree_lines, new_node)
                print(f"New node added at {new_node.state}")

                if self.is_near_goal(new_node):
                    goal_node = self.steer(new_node, self.goal.state)
                    if goal_node and self.is_valid(goal_node.state):
                        goal_node.parent = new_node
                        self.nodes.append(goal_node)
                        self.update_plot(ax, tree_lines, goal_node)

                        print(f"Goal node added at {new_node.state}")
                        best_goal_node = new_node
                        best_cost = new_node.cost
                        break

            if i % 100 == 0:
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
    # obstacles = []
    bounds = np.array(
        [[-2.0, 12.0], [-2.0, 12.0], [-1.0, 1.0], [-1.0, 1.0]]
    )  # [x_min, x_max], [y_min, y_max], [vx_min, vx_max], [vy_min, vy_max]

    rrt = KinodynamicRRTStar(
        start,
        goal,
        obstacles,
        bounds,
        max_iter=1500,
        dt=0.1,
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
