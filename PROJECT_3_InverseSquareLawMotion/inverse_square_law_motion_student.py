"""
平方反比引力场中的运动模拟
文件：inverse_square_law_motion_student.py
作者：李欣欣
日期：2025/6/4
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state_vector
    r = np.sqrt(x**2 + y**2)
    r_cubed = r**3
    
    # 避免除以零
    if r_cubed < 1e-10:
        r_cubed = 1e-10
    
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    
    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]
        t_span (tuple): 积分时间区间 (t_start, t_end)
        t_eval (np.ndarray): 需要存储解的时间点数组
        gm_val (float): GM 值 (引力常数 * 中心天体质量)

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象
    """
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='DOP853',
        rtol=1e-7,
        atol=1e-9
    )
    return sol

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组
        gm_val (float): GM 值
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)

    返回:
        np.ndarray or float: （比）机械能
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        specific_energy = 0.5 * v_squared - gm_val / r
        return specific_energy * m
    else:
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        specific_energy = 0.5 * v_squared - gm_val / r
        return specific_energy * m

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)

    返回:
        np.ndarray or float: （比）角动量
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        specific_Lz = x * vy - y * vx
        return specific_Lz * m
    else:
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        specific_Lz = x * vy - y * vx
        return specific_Lz * m

if __name__ == "__main__":
    # 设置参数
    GM = 1.0
    t_start = 0
    t_end = 20
    t_eval = np.linspace(t_start, t_end, 1000)
    
    # 任务A：不同能量的轨道
    
    # 1. 椭圆轨道 (E < 0)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end), t_eval, GM)
    energy_ellipse = calculate_energy(sol_ellipse.y.T, GM)
    
    # 2. 抛物线轨道 (E ≈ 0)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2)]
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end), t_eval, GM)
    energy_parabola = calculate_energy(sol_parabola.y.T, GM)
    
    # 3. 双曲线轨道 (E > 0)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.5]
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end), t_eval, GM)
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, GM)
    
    # 绘制不同能量的轨道
    plt.figure(figsize=(10, 8))
    plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], label=f'Ellipse (E={energy_ellipse[0]:.3f})')
    plt.plot(sol_parabola.y[0], sol_parabola.y[1], label=f'Parabola (E={energy_parabola[0]:.3f})')
    plt.plot(sol_hyperbola.y[0], sol_hyperbola.y[1], label=f'Hyperbola (E={energy_hyperbola[0]:.3f})')
    plt.plot(0, 0, 'ro', markersize=10, label='Central Mass')
    plt.title('Orbits with Different Energies')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # 任务B：不同角动量的椭圆轨道
    
    # 固定能量 E = -0.5
    E = -0.5
    r_p = 0.5  # 近心点距离
    
    # 计算不同角动量对应的初始速度
    v_p1 = np.sqrt(2*(E + GM/r_p))  # 最大角动量
    v_p2 = 0.8 * v_p1  # 中等角动量
    v_p3 = 0.6 * v_p1  # 小角动量
    
    # 初始条件
    ic_ellipse1 = [r_p, 0.0, 0.0, v_p1]
    ic_ellipse2 = [r_p, 0.0, 0.0, v_p2]
    ic_ellipse3 = [r_p, 0.0, 0.0, v_p3]
    
    # 求解轨道
    sol_ellipse1 = solve_orbit(ic_ellipse1, (t_start, t_end), t_eval, GM)
    sol_ellipse2 = solve_orbit(ic_ellipse2, (t_start, t_end), t_eval, GM)
    sol_ellipse3 = solve_orbit(ic_ellipse3, (t_start, t_end), t_eval, GM)
    
    # 计算角动量
    Lz1 = calculate_angular_momentum(sol_ellipse1.y.T)
    Lz2 = calculate_angular_momentum(sol_ellipse2.y.T)
    Lz3 = calculate_angular_momentum(sol_ellipse3.y.T)
    
    # 绘制不同角动量的椭圆轨道
    plt.figure(figsize=(10, 8))
    plt.plot(sol_ellipse1.y[0], sol_ellipse1.y[1], label=f'Lz={Lz1[0]:.3f}')
    plt.plot(sol_ellipse2.y[0], sol_ellipse2.y[1], label=f'Lz={Lz2[0]:.3f}')
    plt.plot(sol_ellipse3.y[0], sol_ellipse3.y[1], label=f'Lz={Lz3[0]:.3f}')
    plt.plot(0, 0, 'ro', markersize=10, label='Central Mass')
    plt.title('Elliptic Orbits with Different Angular Momenta (E=-0.5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # 验证能量和角动量守恒
    plt.figure(figsize=(12, 5))
    
    # 能量守恒验证
    plt.subplot(1, 2, 1)
    plt.plot(t_eval, calculate_energy(sol_ellipse1.y.T, GM), label='Energy')
    plt.title('Energy Conservation')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(True)
    
    # 角动量守恒验证
    plt.subplot(1, 2, 2)
    plt.plot(t_eval, calculate_angular_momentum(sol_ellipse1.y.T), label='Angular Momentum')
    plt.title('Angular Momentum Conservation')
    plt.xlabel('Time')
    plt.ylabel('Angular Momentum')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
