import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 로드
data = pd.read_pickle('./correlation/FALCO_FOX_on_FINAL_DESTINATION_0_data.pkl')

# Xp와 Yp 데이터 추출 (numpy array)
Xp = data['X'][:-1]
Yp = data['Y'][1:]

xp_common_labels = [
    'x_vec', 'y_vec', 'x_side', 'y_side', 'distance'
]

xp_labels = [
    'percent', 'facing', 'x', 'y', 'action', 'action_frame', 'is_invulnerable', 'character',
    'jumps_left', 'shield', 'on_ground', 'in_hitstun', 'main_x', 'main_y',
    'c_x', 'c_y', 'a', 'b', 'bx', 'by', 'z', 'l', 'r'
]

op_xp_labels = ["op_"+xpl for xpl in xp_labels]

yp_labels = ['A', 'B', 'X', 'Y', 'Z', 'joystick', 'cstick', 'trigger']

# 전체 Xp의 각 정보와 Yp의 각 첫 5개 정보와의 상관관계 계산
correlation_matrix = np.zeros((Xp.shape[1], Yp.shape[1]))
for i in range(Yp.shape[1]):
    for j in range(Xp.shape[1]):
        correlation_matrix[j, i] = np.corrcoef(Xp[:, j], Yp[:, i])[0, 1]

# 히트맵 시각화
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, xticklabels=yp_labels, yticklabels=xp_common_labels+xp_labels+op_xp_labels, cmap='coolwarm', cbar=True)
plt.xlabel('Yp Elements')
plt.ylabel('Xp Features')
plt.title('Correlation between Xp Features and Yp')
plt.tight_layout()
plt.show()

# correlations = {}
# for i in range(Yp.shape[1]):
#     correlations[f'Correlation with {yp_labels[i]}'] = []
#     for j in range(Xp.shape[1]):
#         correlation = np.corrcoef(Xp[:, j], Yp[:, i])[0, 1]
#         correlations[f'Correlation with {yp_labels[i]}'].append(correlation)

# # 결과 출력
# for key, value in correlations.items():
#     print(f"{key}: {value}")
    
# plt.xlabel('Features')
# plt.ylabel('Correlation')
# plt.title('Correlation between Xp Features and first 5 elements of Yp')
# plt.legend()
# plt.grid(True)
# plt.xticks(ticks=np.arange(len(xp_labels)), labels=xp_labels, rotation=45)
# plt.yticks
# plt.tight_layout()
# plt.show()