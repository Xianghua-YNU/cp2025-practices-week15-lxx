#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 修正版
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Args:
        t (float): Independent variable
        y (array-like): State vector [y1, y2]
    
    Returns:
        list: Derivatives [y1', y2']
    """
    # 确保y是数组形式
    y = np.asarray(y)
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    """
    return np.vstack([y[1], -np.pi*(y[0]+1)/4])


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    """
    # 参数验证
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must be in increasing order")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple of length 2")
    if n_points <= 2:  # 修改为<=2而不是<2
        raise ValueError("n_points must be greater than 2")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    
    u_left, u_right = boundary_conditions
    
    def shooting_function(m):
        sol = solve_ivp(ode_system_shooting, x_span, [u_left, m], 
                       t_eval=np.linspace(x_span[0], x_span[1], n_points))
        return sol.y[0, -1] - u_right
    
    # 初始猜测
    m1, m2 = 0.0, -1.0
    f1, f2 = shooting_function(m1), shooting_function(m2)
    
    # 割线法迭代
    for _ in range(max_iterations):
        if abs(f2) < tolerance:
            break
        
        m_new = m2 - f2 * (m2 - m1) / (f2 - f1)
        m1, m2 = m2, m_new
        f1, f2 = f2, shooting_function(m2)
    
    # 最终解
    x = np.linspace(x_span[0], x_span[1], n_points)
    sol = solve_ivp(ode_system_shooting, x_span, [u_left, m2], t_eval=x)
    
    return sol.t, sol.y[0]

def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    """
    # 初始网格和猜测
    x = np.linspace(x_span[0], x_span[1], n_points)
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(boundary_conditions[0], boundary_conditions[1], n_points)
    
    # 求解BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess, tol=1e-6)
    
    # 在更细的网格上评估解
    x_fine = np.linspace(x_span[0], x_span[1], 100)
    y_fine = sol.sol(x_fine)[0]
    
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp.
    """
    # 使用两种方法求解
    x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # 插值到共同网格进行比较
    y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
    
    # 计算差异
    differences = y_shoot - y_scipy_interp
    max_diff = np.max(np.abs(differences))
    rms_diff = np.sqrt(np.mean(differences**2))
    
    # 创建比较图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, y_shoot, 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy, 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of BVP Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 创建差异图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, differences, 'g-', label='Difference')
    plt.xlabel('x')
    plt.ylabel('Difference')
    plt.title('Difference Between Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'x_shooting': x_shoot,
        'y_shooting': y_shoot,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_diff,
        'rms_difference': rms_diff
    }


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # 运行基本测试
    print("Testing ODE system...")
    t_test = 0.5
    y_test = np.array([1.0, 0.5])
    dydt = ode_system_shooting(t_test, y_test)
    print(f"ODE system (shooting): dydt = {dydt}")
    dydt_scipy = ode_system_scipy(t_test, y_test)
    print(f"ODE system (scipy): dydt = {dydt_scipy}")
    
    print("\nTesting boundary conditions...")
    ya = np.array([1.0, 0.5])
    yb = np.array([1.0, -0.3])
    bc_residual = boundary_conditions_scipy(ya, yb)
    print(f"Boundary condition residuals: {bc_residual}")
    
    # 运行方法比较
    print("\nTesting method comparison...")
    results = compare_methods_and_plot()
    print("Method comparison completed successfully!")
    print(f"Max difference: {results['max_difference']:.2e}")
    print(f"RMS difference: {results['rms_difference']:.2e}")
