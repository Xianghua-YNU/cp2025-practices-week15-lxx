def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    
    改进点：
    1. 增加初始网格点数
    2. 提供更好的初始猜测
    3. 设置适当的容差参数
    """
    # 增加初始网格点数以获得更好的收敛性
    x_initial = np.linspace(0, 5, 50)  # 从11增加到50
    
    # 改进初始猜测 - 使用二次函数满足边界条件
    y_initial = np.zeros((2, len(x_initial)))
    y_initial[0] = 3 * (x_initial/5)**2  # y(x) 的二次初始猜测
    y_initial[1] = 6/25 * x_initial      # y'(x) 的导数
    
    # 调用solve_bvp并设置更严格的容差
    sol = solve_bvp(ode_system_for_solve_bvp, 
                   boundary_conditions_for_solve_bvp, 
                   x_initial, y_initial,
                   tol=1e-6,  # 设置更小的容差
                   max_nodes=1000)  # 允许更多节点
    
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed to converge: {sol.message}")
    
    # 在密集网格上评估解
    x_solution = np.linspace(0, 5, 100)
    y_solution = sol.sol(x_solution)[0]
    
    return x_solution, y_solution
