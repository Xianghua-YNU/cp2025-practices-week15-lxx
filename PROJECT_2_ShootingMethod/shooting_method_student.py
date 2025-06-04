#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：李欣欣
学号：20221180076
完成日期：2025/6/4
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
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    """
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
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
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
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
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input parameters
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Setup domain
    x = np.linspace(x_start, x_end, n_points)
    
    # Initial guess for slope
    m1 = -1.0  # First guess
    y0 = [u_left, m1]  # Initial conditions [u(0), u'(0)]
    
    # Solve with first guess
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]  # u(x_end) with first guess
    
    # Check if first guess is good enough
    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]
    
    # Second guess using linear scaling
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0[1] = m2
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]  # u(x_end) with second guess
    
    # Check if second guess is good enough
    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]
    
    # Iterative improvement using secant method
    for iteration in range(max_iterations):
        # Secant method to find better slope
        if abs(u_end_2 - u_end_1) < 1e-12:
            # Avoid division by zero
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        
        # Solve with new guess
        y0[1] = m3
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]
        
        # Check convergence
        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]
        
        # Update for next iteration
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    
    # If not converged, return best solution with warning
    print(f"Warning: Shooting method did not converge after {max_iterations} iterations.")
    print(f"Final boundary error: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Initial mesh and guess
    x = np.linspace(x_span[0], x_span[1], n_points)
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(boundary_conditions[0], boundary_conditions[1], n_points)
    
    # Solve BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess, tol=1e-6)
    
    # Evaluate solution on finer grid
    x_fine = np.linspace(x_span[0], x_span[1], 100)
    y_fine = sol.sol(x_fine)[0]
    
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    """
    # Solve using both methods
    x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # Interpolate to common grid for comparison
    y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
    
    # Calculate differences
    differences = y_shoot - y_scipy_interp
    max_diff = np.max(np.abs(differences))
    rms_diff = np.sqrt(np.mean(differences**2))
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, y_shoot, 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy, 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of BVP Solutions')
    plt.legend()
    plt.grid(True)
    
    # Create difference plot
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


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Try to run comparison
    print("\nTesting method comparison...")
    results = compare_methods_and_plot()
    print("Method comparison completed successfully!")
    print(f"Max difference: {results['max_difference']:.2e}")
    print(f"RMS difference: {results['rms_difference']:.2e}")
