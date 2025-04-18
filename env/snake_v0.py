import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import time
from matplotlib import patches


InitLength = 3
max_food = 1
class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, grid_size = 5, max_steps=1000):
        super(SnakeEnv, self).__init__()

        # 定义状态空间：10x10的网格
        self.grid_size = grid_size

        # 定义动作空间：上、右、下、左
        self.action_space = spaces.Discrete(4)

        # 定义观察空间：10x10的网格，每个格子有4种可能的状态
        # 0: 空, 1: 食物, 2: 蛇头, 3: 蛇身
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(self.grid_size * self.grid_size,),
            dtype=np.uint8
        )
        # 初始化参数
        self.snake = None
        self.food_pos = None
        self.current_direction = None
        self.steps = None
        self.init_length = InitLength
        self.max_steps = max_steps
        self.max_food = max_food

        self.fig = None
        self.ax = None


    # 获取当前环境状态
    def _get_observation(self):
        # 创建空白网格
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # 标记食物的位置
        if self.food_pos is not None:
            grid[self.food_pos[0], self.food_pos[1]] = 1

        # 在网格上标记蛇的位置
        for i, pos in enumerate(self.snake):
            if i==0:
                grid[pos[0], pos[1]] = 2
            else:
                grid[pos[0], pos[1]] = 3
        return grid.flatten()

    # 在空白位置随机放置食物
    def _generate_food(self):
        all_pos = set((i,j)for i in range(self.grid_size) for j in range (self.grid_size))
        snake_pos = set(self.snake)
        empty_grid = all_pos - snake_pos
        if empty_grid:
            num = np.random.choice(len(empty_grid))
            self.food_pos = list(empty_grid)[num]
            return self.food_pos
        else:
            return None

    def reset(self, seed=None, options=None):
        # 重置环境到初始状态
        super().reset(seed=seed)

        # 初始化蛇的位置（初始长度为3）
        self.snake = []
        init_x = self.grid_size//2
        init_y = self.grid_size//2

        for i in range(InitLength):
            self.snake.append((init_x-i,init_y))

        self.current_direction = np.random.randint(0,4)

        # 放置食物
        self.food_pos = self._generate_food()

        # 初始化其他变量
        self.steps = 0
        self.done = False
        self.reward = 0

        info = {}
        return self._get_observation(), info

    # 取反方向
    def _get_opposite(self, action):
        opposites = {0:2, 1:3, 2:0, 3:1}
        return opposites[action]

    def step(self, action):
        self.steps += 1
        reward = 0

        # 处理无效动作
        if action == self._get_opposite(self.current_direction):
            action = self.current_direction  # 保持原方向

        self.current_direction = action

        # 获取当前蛇头位置
        head_x, head_y = self.snake[0]

        # 根据动作计算新的蛇头位置
        if action == 0:  # 上
            new_head = (head_x, head_y - 1)
        elif action == 1:  # 右
            new_head = (head_x + 1, head_y)
        elif action == 2:  # 下
            new_head = (head_x, head_y + 1)
        elif action == 3:  # 左
            new_head = (head_x - 1, head_y)

        # 检查是否终止游戏
        terminated = False
        # 这里不使用
        truncated = False

        # 检查是否撞墙
        if (new_head[0] < 0 or new_head[0] > self.grid_size-1
                or new_head[1] < 0 or new_head[1] > self.grid_size-1):
            # reward -= 15  # 撞墙惩罚
            terminated = True
            # print("撞墙")
            return self._get_observation(), reward, terminated, truncated, {}

        # 检查是否撞到自己
        elif new_head in self.snake[:-1]:
            # reward -= 15  # 撞到自己惩罚
            terminated = True
            # print("撞身")
            return self._get_observation(), reward, terminated, truncated, {}

        # 插入新蛇头
        self.snake.insert(0, new_head)
        # 检查是否吃到食物
        if self.food_pos is not None and new_head == self.food_pos:
            # print("吃到食物")
            reward += 10
            self.food_pos = self._generate_food()
        else:
            # 没吃到食物，移除蛇尾
            self.snake.pop()
            # reward = -0.01  # 小惩罚鼓励蛇尽快找到食物

        if self.food_pos is None:
            reward += 100
            terminated = True
            return self._get_observation(), reward, terminated, truncated, {}

        # # 计算距离奖励
        # old_head = self.snake[0]
        # foods = np.argwhere(self.grid == 1)
        # if len(foods) == 0:
        #     # 没有食物，可能是游戏已终止
        #     food_pos = (-1, -1)  # 默认值
        # else:
        #     food_pos = foods[0]
        # old_dist = abs(old_head[0] - food_pos[0]) + abs(old_head[1] - food_pos[1])
        #
        # # 移动后计算新距离
        # new_dist = abs(new_head[0] - food_pos[0]) + abs(new_head[1] - food_pos[1])
        # reward += (old_dist - new_dist) * 0.2  # 距离缩短给予奖励

        # 强制终止条件——超过最大步数
        if self.steps > self.max_steps:
            terminated = True
            reward = -5  # 负奖励促使蛇加快行动
            return self._get_observation(), reward, terminated, truncated, {}

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        # 获取当前网格状态并重塑为二维数组
        grid = self._get_observation().reshape((self.grid_size, self.grid_size))

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            self.ax.set_xticks(np.arange(self.grid_size + 1))
            self.ax.set_yticks(np.arange(self.grid_size + 1))
            self.ax.grid(True, linestyle='-', linewidth=2)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.invert_yaxis()
        # 清空画布
        for patch in self.ax.patches[:]:
            patch.remove()

        # 绘制网格线
        for i in range(self.grid_size + 1):
            self.ax.axhline(i, color='black', linewidth=2)
            self.ax.axvline(i, color='black', linewidth=2)

        # 绘制食物
        if self.food_pos is not None:
            food = patches.Rectangle(
                (self.food_pos[0], self.food_pos[1]), 1, 1,
                facecolor='red', edgecolor='black', alpha=0.7
            )
            self.ax.add_patch(food)

        # 绘制蛇身
        for i, pos in enumerate(self.snake):
            color = 'green' if i == 0 else 'blue'
            body = patches.Rectangle(
                (pos[0], pos[1]), 1, 1,
                facecolor=color, edgecolor='black', alpha=0.7
            )
            self.ax.add_patch(body)

        # 更新显示
        plt.title(f"Steps: {self.steps}")
        plt.pause(0.3)

        if self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def close(self):
        """关闭渲染窗口并释放资源"""
        if hasattr(self, 'fig'):
            # 关闭Matplotlib图形窗口
            plt.close(self.fig)
            # 删除引用以释放内存
            del self.fig
            del self.ax


if __name__ == "__main__":
    env = SnakeEnv()
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        print(f"action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            break

    env.close()