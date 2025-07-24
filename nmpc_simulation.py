import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 参数定义
N = 10          # 控制/预测步长
T = 0.2         # 每步采样时间
L = 2.0         # 车辆长度

# 状态与控制输入
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.size()[0]

v = ca.SX.sym('v')       # 速度
delta = ca.SX.sym('delta')  # 转向角
controls = ca.vertcat(v, delta)
n_controls = controls.size()[0]

# 非线性动力学模型
rhs = ca.vertcat(
    v * ca.cos(theta),
    v * ca.sin(theta),
    v / L * ca.tan(delta)
)
f = ca.Function('f', [states, controls], [rhs])

# 迭代次数
k_steps = 100
n = 3
p = 2
# 开辟所有状态x的存储空间并初始状态
X_k = np.zeros((n, k_steps+1))

# 开辟所有控制输入u的存储空间
U_k = np.zeros((p, k_steps))

# 状态预测函数（Euler法 离散化）
# def state_model(x0, u_seq):
#     x_traj = [x0]
#     x_current = x0
#     for i in range(N):
#         u_current = u_seq[:, i]
#         x_current = x_current + T * f(x_current, u_current)
#         x_traj.append(x_current)
#     return ca.hcat(x_traj)
def system_model(x_current, u_current):
    x_current = x_current + T * f(x_current, u_current)
    return x_current

def state_model(x0, u_seq):
    x_traj = [x0]
    x_current = x0
    for i in range(N):
        u_current = u_seq[:, i]
        # x_current = x_current + T * f(x_current, u_current)
        x_current = system_model(x_current, u_current)
        x_traj.append(x_current)

    return ca.hcat(x_traj)
def nmpc_predict(x_c, U, Q, R):
    # 优化变量
    # U = ca.SX.sym('U', n_controls, N)
    X_traj = state_model(x_c, U)
    # X_k[:, 0] = X_traj
    # 成本函数
    # Q = np.diag([10, 10, 1])  # 状态误差权重
    # R = np.diag([1, 10])  # 控制输入权重
    obj = 0
    for k in range(N):
        dx = X_traj[:, k] - ref[:, k]
        obj += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])

    # 约束（例如控制限）
    lbg = []
    ubg = []
    g = []
    for k in range(N):
        g.append(U[0, k])  # v
        g.append(U[1, k])  # delta
        lbg += [0.0, -0.5]  # v in [0, 1.5], delta in [-0.5, 0.5] rad
        ubg += [1.5, 0.5]

    # 构造 NLP 求解器
    opt_vars = ca.reshape(U, -1, 1)
    nlp_prob = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob)

    # 初始化控制输入猜测
    u0_guess = np.zeros((n_controls, N))
    u0_flat = u0_guess.reshape((-1, 1))

    # 求解
    sol = solver(x0=u0_flat, lbg=lbg, ubg=ubg)
    u_opt = ca.reshape(sol['x'], n_controls, N)
    return u_opt

if __name__ == "__main__":
    # 初始状态与目标轨迹
    x0 = ca.DM([0, 2, 0])  # 初始位置
    ref = np.array([[i*0.5, 0, 0] for i in range(k_steps+1)]).T  # 目标：向x轴方向直行

    # 优化变量
    U = ca.SX.sym('U', n_controls, N)
    # 成本函数
    Q = np.diag([10, 10, 1])     # 状态误差权重
    R = np.diag([1, 10])         # 控制输入权重
    for i in range(k_steps):
        X_k[:, i] = X_traj
        u_opt = nmpc_predict()




    # 输出结果
    print("Optimal control sequence:")
    print(u_opt)




    # ---------------------------
    # 可视化：预测轨迹 vs 参考轨迹
    # ---------------------------
    # X_pred = predict_state(x0, u_opt)
    # X_pred_val = ca.Function('X_val', [], [X_pred])()['o0'].full()
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(ref[0, :], ref[1, :], 'r--', label='Reference Path')              # 目标轨迹
    # plt.plot(X_pred_val[0, :], X_pred_val[1, :], 'b.-', label='Predicted Path')  # 预测轨迹
    #
    # plt.xlabel('X position')
    # plt.ylabel('Y position')
    # plt.title('NMPC Predicted Trajectory')
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()