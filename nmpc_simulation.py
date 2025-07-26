import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# === 参数定义 ===
N = 10          # 控制/预测步长
T = 0.2         # 采样周期
L = 2.0         # 车辆轴距
k_steps = 100   # 模拟迭代次数

# === 状态和控制输入符号变量 ===
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.size()[0]

v = ca.SX.sym('v')
delta = ca.SX.sym('delta')
controls = ca.vertcat(v, delta)
n_controls = controls.size()[0]

# === 非线性模型 ===
rhs = ca.vertcat(
    v * ca.cos(theta),
    v * ca.sin(theta),
    v / L * ca.tan(delta)
)
f = ca.Function('f', [states, controls], [rhs])

# === 状态预测函数 ===
def system_model(x_current, u_current):
    return x_current + T * f(x_current, u_current)

def state_model(x0, u_seq):
    x_traj = [x0]
    x_current = x0
    for i in range(N):
        u_current = u_seq[:, i]
        x_current = system_model(x_current, u_current)
        x_traj.append(x_current)
    return ca.hcat(x_traj)

# === NMPC 优化器构建函数 ===
def nmpc_optimize(x_init, ref, Q, R):
    U = ca.SX.sym('U', n_controls, N)  # 符号优化变量
    X_pred = state_model(x_init, U)

    obj = 0
    for k in range(N):
        dx = X_pred[:, k] - ref[:, k]
        obj += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])

    # 控制输入约束
    g = []
    lbg = []
    ubg = []
    for k in range(N):
        g += [U[0, k], U[1, k]]
        lbg += [0.0, -0.5]    # v ∈ [0, 1.5], delta ∈ [-0.5, 0.5]
        ubg += [1.5, 0.5]

    opt_vars = ca.reshape(U, -1, 1)
    nlp_prob = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob)

    # 初始猜测
    u0_guess = np.zeros((n_controls * N, 1))
    sol = solver(x0=u0_guess, lbg=lbg, ubg=ubg)

    u_opt = ca.reshape(sol['x'], n_controls, N)
    return u_opt.full()  # 转为NumPy数组返回

# === 主程序 ===
if __name__ == "__main__":
    X_k = np.zeros((n_states, k_steps + 1))
    U_k = np.zeros((n_controls, k_steps))
    X_k[:, 0] = np.array([0, 2, 0])  # 初始状态

    Q = np.diag([10, 10, 1])
    R = np.diag([1, 10])

    for i in range(k_steps):
        # 目标轨迹向 x 轴方向移动
        ref = np.array([[X_k[0, i] + j * 0.5, 0, 0] for j in range(N + 1)]).T

        x_k = ca.DM(X_k[:, i])  # 当前状态作为 casadi DM 类型传入
        u_traj = nmpc_optimize(x_k, ref, Q, R)  # 返回控制序列（NumPy数组）

        u_k = u_traj[:, 0]  # 取第一步控制应用
        x_next = system_model(x_k, ca.DM(u_k))

        X_k[:, i + 1] = np.array(x_next.full()).flatten()
        U_k[:, i] = u_k

    # === 可视化 ===
    plt.figure(figsize=(8, 6))
    plt.plot(X_k[0, :], X_k[1, :], label="NMPC Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Nonlinear MPC Path Tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
