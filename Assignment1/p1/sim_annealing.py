import numpy as np
import math
import random


with open('points.txt', 'r') as f:
    points = [tuple(map(float, line.split())) for line in f if line.strip()]

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def iteration(points_list):
    num_points=len(points_list)
    start_point=random.randint(0,num_points-1)

    touched_points=np.zeros(num_points)
    touched_points[start_point]=1
    
    route=[start_point]
    total_dist=0
    current=start_point

    while len(route)<num_points:
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


