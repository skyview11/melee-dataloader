import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# 주어진 조이스틱 위치
joystick_positions = [
    (0, 0), (0.0, 0.5), (0, 1), 
    (0.3535533905932738, 0.3535533905932738), (0.7071067811865476, 0.7071067811865476), 
    (0.5, 0.0), (1, 0), (0.3535533905932738, -0.3535533905932738), 
    (0.7071067811865476, -0.7071067811865476), (0.0, -0.5), (0, -1), 
    (-0.3535533905932738, -0.3535533905932738), (-0.7071067811865476, -0.7071067811865476), 
    (-0.5, 0.0), (-1, 0), (-0.3535533905932738, 0.3535533905932738), 
    (-0.7071067811865476, 0.7071067811865476)
]

# KDTree를 사용하여 가장 가까운 점 찾기
tree = KDTree(joystick_positions)

# [-1, 1] x [-1, 1] 영역에 대한 격자 점 생성
grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

# 각 격자 점에 대해 가장 가까운 조이스틱 위치 찾기
distances, indices = tree.query(grid_points)
closest_points = np.array(joystick_positions)[indices]

# 분류 결과를 시각화
plt.figure(figsize=(8, 8))
plt.scatter(grid_points[:, 0], grid_points[:, 1], c=indices, cmap='tab20', marker='.', alpha=0.5)
plt.scatter(*zip(*joystick_positions), color='red', edgecolor='black', s=100)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Classification of Points in [-1, 1] x [-1, 1] by Nearest Joystick Position')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
