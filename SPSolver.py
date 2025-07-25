import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import deque
import time

class VRPProblem:

    def __init__(self,coords, dist_matrix, capacities, demands):
        self.depot_index = 0
        self.coords = coords
        self.dist_matrix = dist_matrix
        self.capacities = capacities
        self.demands = demands
        self.num_vehicles = len(capacities)
        self.num_nodes = len(demands)
        
    def __str__(self):
        return (f"VRP Problem: {self.num_nodes} nodes ({self.num_vehicles} vehicles)\n"
                f"Total demand: {sum(self.demands[1:]):.1f}, "
                f"Total capacity: {sum(self.capacities):.1f}")

class VRPSolution:
    def __init__(self, problem, total_cost, routes):
        self.problem = problem
        self.total_cost = total_cost
        self.routes = routes
        
    def __str__(self):
        s = f"VRP Solution (Total Cost: {self.total_cost:.2f})\n"
        for i, route in enumerate(self.routes):
            s += f"Vehicle {i+1} ({self.problem.capacities[i]}): {route}\n"
        return s

# =============================
# SOLVER CLASS
# =============================
class ClassicalSPSolver:
    def __init__(self, num_permutations=100):
        self.num_permutations = num_permutations
        
    def solve_tsp(self, dist_matrix):
        """Solve TSP using nearest neighbor heuristic"""
        n = dist_matrix.shape[0]
        unvisited = set(range(n))
        current = 0  # Start at depot
        path = [current]
        unvisited.remove(current)
        total_cost = 0
        
        while unvisited:
            # Find nearest unvisited neighbor
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            total_cost += dist_matrix[current][nearest]
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Return to depot
        total_cost += dist_matrix[current][0]
        path.append(0)
        return path, total_cost
    
    def _precompute_segment_costs(self, tour, dist_matrix, depot_index):
        """Precompute costs for all segments with depot connections"""
        n = len(tour)
        seg_cost = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Start: depot to first node
                start_cost = dist_matrix[depot_index, tour[i]]
                
                # Middle: path between nodes
                mid_cost = 0
                for k in range(i, j):
                    mid_cost += dist_matrix[tour[k], tour[k+1]]
                
                # End: last node to depot
                end_cost = dist_matrix[tour[j], depot_index]
                
                seg_cost[i, j] = start_cost + mid_cost + end_cost
                
        return seg_cost
    
    def solve(self, problem):
        """Main solver function"""
        print("Starting VRP solver...")
        start_time = time.time()
        
        # Phase 1: Solve TSP
        tsp_start = time.time()
        tsp_tour, tsp_cost = self.solve_tsp(problem.dist_matrix)
        print(f"  TSP solved in {time.time()-tsp_start:.2f}s (Cost: {tsp_cost:.2f})")
        
        # Remove depot from tour (if present)
        if problem.depot_index in tsp_tour:
            tsp_tour = [n for n in tsp_tour if n != problem.depot_index]
        
        # Phase 2: Solution Partitioning
        sps_start = time.time()
        best_solution = None
        
        # Precompute segment costs and demands
        seg_cost = self._precompute_segment_costs(
            tsp_tour, problem.dist_matrix, problem.depot_index
        )
        
        # Cumulative demands
        cum_demand = np.zeros(len(tsp_tour)+1)
        for i in range(1, len(cum_demand)):
            cum_demand[i] = cum_demand[i-1] + problem.demands[tsp_tour[i-1]]
        
        # Try multiple vehicle permutations
        for perm_idx in range(self.num_permutations):
            # Random vehicle permutation
            vehicle_perm = np.random.permutation(problem.capacities)
            
            # DP initialization
            dp = np.full((problem.num_vehicles+1, len(tsp_tour)+1), float('inf'))
            dp[0, 0] = 0  # 0 vehicles, 0 customers
            
            # Backpointer for route reconstruction
            back = np.zeros((problem.num_vehicles+1, len(tsp_tour)+1), dtype=int)
            
            # Dynamic programming with monotonic queue
            for k in range(1, problem.num_vehicles+1):
                cap = vehicle_perm[k-1]
                Q = deque()
                total_delta = 0
                left = 0  # Left pointer for capacity constraint
                
                for i in range(1, len(tsp_tour)+1):
                    # Update cost delta for segment [j, i]
                    if i > 1:
                        delta = (
                            problem.dist_matrix[tsp_tour[i-2], tsp_tour[i-1]] +
                            problem.dist_matrix[tsp_tour[i-1], problem.depot_index] -
                            problem.dist_matrix[tsp_tour[i-2], problem.depot_index]
                        )
                        total_delta += delta
                    
                    # Add new candidate j = i-1
                    candidate = dp[k-1, i-1] + seg_cost[i-1, i-1]
                    candidate_val = candidate - total_delta
                    
                    # Maintain monotonic queue
                    while Q and Q[-1][1] > candidate_val:
                        Q.pop()
                    Q.append((i-1, candidate_val))
                    
                    # Update capacity constraint
                    while cum_demand[i] - cum_demand[left] > cap:
                        left += 1
                    
                    # Remove invalid candidates
                    while Q and Q[0][0] < left:
                        Q.popleft()
                    
                    # Update DP state
                    if Q:
                        dp[k, i] = Q[0][1] + total_delta
                        back[k, i] = Q[0][0]
            
            # Check if we found a better solution
            total_cost = dp[problem.num_vehicles, len(tsp_tour)]
            if best_solution is None or total_cost < best_solution.total_cost:
                # Reconstruct routes
                routes = []
                current = len(tsp_tour)
                for k in range(problem.num_vehicles, 0, -1):
                    start = back[k, current]
                    if start < current:  # Non-empty route
                        route_nodes = tsp_tour[start:current]
                        routes.append(
                            [problem.depot_index] + route_nodes + [problem.depot_index]
                        )
                    else:
                        routes.append([problem.depot_index, problem.depot_index])
                    current = start
                routes.reverse()
                best_solution = VRPSolution(problem, total_cost, routes)
        
        print(f"  SPS solved in {time.time()-sps_start:.2f}s "
              f"({self.num_permutations} permutations)")
        print(f"Total solve time: {time.time()-start_time:.2f}s")
        return best_solution