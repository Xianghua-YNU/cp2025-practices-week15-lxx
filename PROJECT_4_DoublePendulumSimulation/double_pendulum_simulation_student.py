import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 常量定义
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4   # 每个摆臂的长度 (m)
M_CONST = 1.0   # 每个摆锤的质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g):
    """
    返回双摆状态向量y的时间导数。
    
    参数:
        y: 当前状态向量 [theta1, omega1, theta2, omega2]
        t: 当前时间 (odeint 需要)
        L1, L2: 摆臂长度
        m1, m2: 摆锤质量
        g: 重力加速度
        
    返回:
        时间导数 [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]
    """
    theta1, omega1, theta2, omega2 = y
    
    # 计算domega1_dt和domega2_dt的分母
    common_denominator = 3 - np.cos(2*theta1 - 2*theta2)
    
    # 计算domega1_dt的分子
    domega1_dt_numerator = - (omega1**2 * np.sin(2*theta1 - 2*theta2) +
                              2 * omega2**2 * np.sin(theta1 - theta2) +
                              (g/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1)))
    domega1_dt = domega1_dt_numerator / common_denominator
    
    # 计算domega2_dt的分子
    domega2_dt_numerator = (4 * omega1**2 * np.sin(theta1 - theta2) +
                            omega2**2 * np.sin(2*theta1 - 2*theta2) +
                            2 * (g/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2)))
    domega2_dt = domega2_dt_numerator / common_denominator
    
    return [omega1, domega1_dt, omega2, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    使用 odeint 求解双摆的常微分方程组。
    
    参数:
        initial_conditions: 初始条件字典 {'theta1': , 'omega1': , 'theta2': , 'omega2': }
        t_span: 时间范围 (t_start, t_end)
        t_points: 时间点数量
        L_param: 摆臂长度
        g_param: 重力加速度
        
    返回:
        t_arr: 时间数组
        sol_arr: 解数组 [theta1, omega1, theta2, omega2] 随时间变化
    """
    # 从字典创建初始状态向量
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'],
          initial_conditions['theta2'], initial_conditions['omega2']]
    
    # 创建时间数组
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # 调用odeint求解微分方程
    sol_arr = odeint(derivatives, y0, t_arr, 
                     args=(L_param, L_param, M_CONST, M_CONST, g_param),
                     rtol=1e-8, atol=1e-8)
    
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统的总能量 (动能 + 势能)。
    
    参数:
        sol_arr: 解数组 [theta1, omega1, theta2, omega2]
        L_param: 摆臂长度
        m_param: 摆锤质量
        g_param: 重力加速度
        
    返回:
        总能量数组
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]
    
    # 计算势能
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))
    
    # 计算动能
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + 
                                omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    创建双摆运动的动画。
    
    参数:
        t_arr: 时间数组
        sol_arr: 解数组
        L_param: 摆臂长度
        skip_frames: 跳帧数
        
    返回:
        动画对象
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]
    
    # 为动画选择帧
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]
    
    # 转换为笛卡尔坐标
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)
    
    # 创建图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False, 
                         xlim=(-2*L_param - 0.1, 2*L_param + 0.1), 
                         ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')
    
    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='red')
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % t_anim[i])
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                  interval=25, blit=True, init_func=init)
    return ani

if __name__ == '__main__':
    # 初始条件 (角度单位为弧度)
    initial_conditions = {
        'theta1': np.pi/2,  # 90度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90度
        'omega2': 0.0
    }
    
    # 模拟参数
    t_start = 0
    t_end = 10  # 模拟时间
    t_points = 1000  # 时间点数
    
    # 求解双摆运动
    t, sol = solve_double_pendulum(initial_conditions, (t_start, t_end), t_points)
    
    # 计算能量
    energy = calculate_energy(sol)
    
    # 绘制能量随时间变化
    plt.figure(figsize=(10, 5))
    plt.plot(t, energy)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (J)')
    plt.title('Double Pendulum Energy Conservation')
    plt.grid(True)
    plt.show()
    
    # 计算能量变化
    energy_variation = np.max(energy) - np.min(energy)
    print(f"Initial energy: {energy[0]:.7f} J")
    print(f"Energy variation: {energy_variation:.2e} J")
    
    # 创建动画
    ani = animate_double_pendulum(t, sol, skip_frames=10)
    plt.show()
