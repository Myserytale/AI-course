import numpy as np
import random
import math

with open('points.txt', 'r') as f:
    points = [tuple(map(float, line.split())) for line in f if line.strip()]

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def interation(points_list, start=None):
    num_points=len(points_list)

    if start is None:
        start_point = random.randint(0, num_points - 1)
    else:
        start_point = start
    
    touched_points = np.zeros(num_points)
    touched_points[start_point] = 1
    
    route = [start_point] 
    total_dist = 0
    current = start_point

    while len(route) < num_points:
        min_d=float('inf')
        next_p=-1

        for i in range(num_points):
            if touched_points[i]==0:
                d=dist(points_list[current],points_list[i])
                if d < min_d:
                    min_d=d
                    next_p=i

        touched_points[next_p]=1
        route.append(next_p)
        total_dist+=min_d
        current=next_p

    total_dist+=dist(points_list[current],points_list[start_point])
    return route, total_dist

def main():
    num_points = len(points)
    if num_points == 0:
        print("No points loaded.")
        return

    # 1. Get our initial solution (route and distance)
    current_route, current_dist = interation(points)
    
    best_route = current_route[:]
    best_dist = current_dist
    
    # 2. Changed 1D tabu_list to a 2D matrix. 
    # Tabu search for TSP tracks pairs of swapped cities, which requires an N x N grid.
    tabu_matrix = np.zeros((num_points, num_points))
    
    # Tabu Search parameters (you can tweak these)
    max_iterations = 1000
    tabu_tenure = 10     

    print(f"Initial Distance: {best_dist:.2f}")

    # 3. Main Tabu Search Loop
    for it in range(max_iterations):
        best_neighbor_route = None
        best_neighbor_dist = float('inf')
        best_swap = None

        # Explore the neighborhood using "2-opt" swaps (reversing route segments)
        for i in range(num_points - 1):
            for j in range(i + 1, num_points):
                # Check if this specific swap is currently on the Tabu list
                is_tabu = tabu_matrix[i][j] > it

                # Create neighbor by reversing the order of nodes between i and j
                neighbor_route = current_route[:]
                neighbor_route[i:j+1] = reversed(neighbor_route[i:j+1])

                # Calculate the total distance of this new neighbor route
                ndist = 0
                for k in range(num_points):
                    ndist += dist(points[neighbor_route[k]], points[neighbor_route[(k+1)%num_points]])

                # Aspiration criterion: if the move is Tabu, but it results in a new 
                # all-time global best, we ignore the Tabu status and take it anyway.
                if ndist < best_neighbor_dist and (not is_tabu or ndist < best_dist):
                    best_neighbor_dist = ndist
                    best_neighbor_route = neighbor_route
                    best_swap = (i, j)

        # 4. Move to the best valid neighbor found in this iteration
        if best_neighbor_route is not None:
            current_route = best_neighbor_route
            current_dist = best_neighbor_dist

            # Put the swap we just made onto the Tabu list for 'tabu_tenure' iterations
            tabu_matrix[best_swap[0]][best_swap[1]] = it + tabu_tenure
            tabu_matrix[best_swap[1]][best_swap[0]] = it + tabu_tenure

            # Update the global best if we beat it
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = current_route[:]
                print(f"Iteration {it}: New best distance = {best_dist:.2f}")

    print("\nTabu Search Complete!")
    print(f"Final Best Distance: {best_dist:.2f}")

if __name__ == "__main__":
    main()
