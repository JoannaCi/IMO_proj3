import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import time
import multiprocessing as mp
from copy import deepcopy

def score(cities, paths):
    cycle_1 = paths[0] + [paths[0][0]]
    cycle_2 = paths[1] + [paths[1][0]]
    score_1=sum(cities[cycle_1[i], cycle_1[i+1]] for i in range(len(cycle_1) - 1))
    score_2=sum(cities[cycle_2[i], cycle_2[i+1]] for i in range(len(cycle_2) - 1))

    return score_1+score_2

def delta_insert(cities, path, i, city):
    a, b = path[i - 1], path[i]
    return cities[a, city] + cities[city, b] - cities[a, b]

def delta_replace_vertex(cities, path, i, city):
    path_len = len(path)
    a, b, c = path[(i - 1)%path_len], path[i], path[(i+1)%path_len]
    return cities[a, city] + cities[city, c] - cities[a, b] - cities[b, c]

def delta_replace_vertices_outside(cities, paths, i, j):
    return delta_replace_vertex(cities, paths[0], i, paths[1][j]) + delta_replace_vertex(cities, paths[1], j, paths[0][i])

def delta_replace_vertices_inside(cities, path, i, j):
    path_len = len(path)
    a, b, c = path[(i - 1)%path_len], path[i], path[(i+1)%path_len]
    d, e, f = path[(j-1)%path_len], path[j], path[(j+1)%path_len]
    if j-i == 1:
        return cities[a,e]+cities[b,f]-cities[a,b]-cities[e,f]
    elif (i, j) == (0, len(path)-1):
        return cities[e, c] + cities[d, b] - cities[b, c] - cities[d, e]
    else:
        return cities[a,e] + cities[e,c] + cities[d,b] + cities[b,f] -cities[a,b]-cities[b,c]-cities[d,e] - cities[e,f] 

def delta_replace_edges_inside(cities, path, i, j):
    path_len = len(path)
    if (i, j) == (0, len(path)-1):
        a, b, c, d = path[i], path[(i+1)%path_len], path[(j-1)%path_len], path[j]
    else:
        a, b, c, d = path[(i - 1)%path_len], path[i], path[j], path[(j+1)%path_len]
    return cities[a, c] + cities[b, d] - cities[a, b] - cities[c, d]

def outside_candidates(paths):
    indices = list(range(len(paths[0]))), list(range(len(paths[1])))
    indices_pairwise = list(itertools.product(*indices))
    return indices_pairwise

def inside_candidates(path):
    combinations = []
    for i in range(len(path)):
        for j in range(i+1, len(path)):
            combinations.append([i, j])
    return combinations

def replace_vertices_outside(paths, i, j):
    temp = paths[0][i]
    paths[0][i] = paths[1][j]
    paths[1][j] = temp

def replace_vertices_inside(path, i, j):
    temp = path[i]
    path[i] = path[j]
    path[j] = temp
    
def replace_edges_inside(path, i, j):
    if (i, j) == (0, len(path)-1):
        temp = path[i]
        path[i] = path[j]
        path[j] = temp     
    path[i:j+1] = reversed(path[i:j+1])
    
def regret(args):
    cities, start_idx = args
    n = cities.shape[0]
    unvisited = list(range(n))
    
    tour1 = [unvisited.pop(start_idx)]
    nearest_to_first_1 = [cities[tour1[0]][j] for j in unvisited]
    tour1.append(unvisited.pop(np.argmin(nearest_to_first_1)))

    start_city_2_idx = np.argmax([cities[tour1[0]][i] for i in unvisited])
    tour2 = [unvisited.pop(start_city_2_idx)]

    nearest_to_first_2 = [cities[tour2[0]][j] for j in unvisited]
    tour2.append(unvisited.pop(np.argmin(nearest_to_first_2)))

    nearest_to_tour_1 = [cities[tour1[0]][j] + cities[tour1[1]][j] for j in unvisited]
    tour1.append(unvisited.pop(np.argmin(nearest_to_tour_1)))

    nearest_to_tour_2 = [cities[tour2[0]][j] + cities[tour2[1]][j] for j in unvisited]
    tour2.append(unvisited.pop(np.argmin(nearest_to_tour_2)))

    while len(unvisited) > 0:
        for tour in [tour1, tour2]:
            regrets = []
            for city in unvisited:
                distances = [cities[tour[i]][city] + cities[city][tour[i+1]] - cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
                distances.append(cities[tour[0]][city] + cities[city][tour[-1]] - cities[tour[-1]][tour[0]])
                distances.sort()
                regret = distances[1] - distances[0]
                regret -= 0.37 * distances[0]
                regrets.append((regret, city))
            regrets.sort(reverse=True)
            best_city = regrets[0][1]
            tour_distances = [cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
            best_increase = float('inf')
            best_index = -1
            for i in range(len(tour_distances)):
                increase = cities[best_city][tour[i]] + cities[best_city][tour[i+1]] - tour_distances[i]
                if increase < best_increase:
                    best_increase = increase
                    best_index = i + 1
            tour.insert(best_index, best_city)
            unvisited.remove(best_city)
    return [tour1,tour2]

class Steepest(object):
    def __init__(self, cities):
        self.cities = cities
        self.delta = delta_replace_edges_inside
        self.replace = replace_edges_inside
        self.moves = [self.outside_vertices_trade_best, self.inside_trade_best]
    
    def outside_vertices_trade_best(self, cities, paths):
        candidates = outside_candidates(paths)
        scores = np.array([delta_replace_vertices_outside(cities, paths, i, j) for i, j in candidates])
        best_result_idx = np.argmin(scores)
        if scores[best_result_idx] < 0:
            return replace_vertices_outside, (paths, *candidates[best_result_idx]), scores[best_result_idx]
        return None, None, scores[best_result_idx]
            
    def inside_trade_best(self, cities, paths):
        combinations = inside_candidates(paths[0]), inside_candidates(paths[1])
        scores = np.array([[self.delta(cities, paths[idx], i, j) for i, j in combinations[idx]] for idx in range(len(paths))])
        best_path_idx, best_combination = np.unravel_index(np.argmin(scores), scores.shape)
        best_score = scores[best_path_idx, best_combination]
        if best_score < 0:
            return self.replace, (paths[best_path_idx], *combinations[best_path_idx][best_combination]), best_score
        return None, None, best_score 
    
    def __call__(self, paths):
        paths = deepcopy(paths)
        start = time.time()
        while True:
            replace_funs, args, scores = list(zip(*[move(self.cities, paths) for move in self.moves]))
            best_score_idx = np.argmin(scores)
            if scores[best_score_idx] < 0:
                replace_funs[best_score_idx](*args[best_score_idx])
            else:
                break
        return time.time()-start, paths
    
def pairwise_distances(points):
    num_points = len(points)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])

    return dist_matrix

def plot_optimized_tours(positions, cycle1, cycle2, method):
    cycle1.append(cycle1[0])
    cycle2.append(cycle2[0])

    plt.figure()
    plt.plot(positions[cycle1, 0], positions[cycle1, 1], linestyle='-', marker='o', color='r', label='Cycle 1')
    plt.plot(positions[cycle2, 0], positions[cycle2, 1], linestyle='-', marker='o', color='b', label='Cycle 2')

    plt.legend()
    plt.title(method)
    plt.show()

def index(xs, e):
    try:
        return xs.index(e)
    except:
        return None
        
def find_node(paths, a):
    i = index(paths[0], a)
    if i is not None: return 0, i
    i = index(paths[1], a)
    if i is not None: return 1, i
    print(paths)
    assert False, f'City {a} must be in either cycle'
    
def remove_at(xs, sorted_indices):
    for i in reversed(sorted_indices):
        del(xs[i])

def reverse(xs, i, j):
    n = len(xs)
    d = (j - i) % n
    for k in range(abs(d)//2+1):
        a, b = (i+k)%n, (i+d-k)%n
        xs[a], xs[b] = xs[b], xs[a]

def insert_move(moves, move):
    delta_x = move[0]
    for i, x in enumerate(moves):
        delta_y = x[0]
        if delta_x < delta_y:
            moves.insert(i, move)
            return
        elif delta_x == delta_y:
            return
    moves.append(move)

def has_edge(cycle, a, b):
    for i in range(len(cycle) - 1):
        x, y = cycle[i], cycle[i+1]
        if (a, b) == (x, y): return +1
        if (a, b) == (y, x): return -1
        
    x, y = cycle[-1], cycle[0]
    if (a, b) == (x, y): return +1
    if (a, b) == (y, x): return -1
    return 0

def any_has_edge(paths, a, b):
    for i in range(2):
        status = has_edge(paths[i], a, b)
        if status != 0: return i, status
    return None, 0

def delta_swap_node(D, x1, y1, z1, x2, y2, z2):
    return D[x1,y2] + D[z1,y2] - D[x1,y1] - D[z1,y1] + D[x2,y1] + D[z2,y1] - D[x2,y2] - D[z2,y2]

def make_swap_node(cities, paths, cycycle1, i, cycycle2, j):
    cycle1, cycle2 = paths[cycycle1], paths[cycycle2]
    D = cities
    n, m = len(cycle1), len(cycle2)
    x1, y1, z1 = cycle1[(i-1)%n], cycle1[i], cycle1[(i+1)%n]
    x2, y2, z2 = cycle2[(j-1)%m], cycle2[j], cycle2[(j+1)%m]
    delta = delta_swap_node(cities, x1, y1, z1, x2, y2, z2)
    move = delta, NODE, cycycle1, cycycle2, x1, y1, z1, x2, y2, z2
    return delta, move

def delta_swap_edge(cities, a, b, c, d):
    if a == d or a == b or a == c or b == c or b == d or c == d: return 1e8
    return cities[a, c] + cities[b, d] - cities[a, b] - cities[c, d]

def gen_swap_edge_2(cities, cycle, i, j):
    n = len(cycle)
    nodes = cycle[i], cycle[(i+1)%n], cycle[j], cycle[(j+1)%n]
    return (delta_swap_edge(cities, *nodes), *nodes)

def delta_swap_edge_2(cities, cycle, i, j):
    return gen_swap_edge_2(cities, cycle, i, j)[0]

def gen_swap_edge(n):
    return [(i, (i+d)%n) for i in range(n) for d in range(2, n-1)]

def gen_swap_node(n, m):
    return [(i, j) for i in range(n) for j in range(m)]

def init_moves(cities, paths):
    moves = []
    for k in range(2):
        cycle = paths[k]
        n = len(cycle)
        for i, j in gen_swap_edge(n):
            delta, a, b, c, d = gen_swap_edge_2(cities, cycle, i, j)
            if delta < 0: moves.append((delta, EDGE, a, b, c, d))
    for i, j in gen_swap_node(len(paths[0]), len(paths[1])):
        delta, move = make_swap_node(cities, paths, 0, i, 1, j)
        if delta < 0: moves.append(move)
    return moves

EDGE, NODE = range(2)
def apply_move(paths, move):
    kind = move[1]
    if kind == EDGE:
        _, _, a, _, c, _ = move
        (cycle1, i), (cycle2, j) = find_node(paths, a), find_node(paths, c)
        cycle = paths[cycle1]
        n = len(cycle)
        reverse(cycle, (i+1)%n, j)
    elif kind == NODE:
        _, _, cycle1, cycle2, _, a, _, _, b, _ = move
        i, j = paths[cycle1].index(a), paths[cycle2].index(b)
        paths[cycle1][i], paths[cycle2][j] = paths[cycle2][j], paths[cycle1][i]
    else:
        assert False, 'Invalid move type'

class SearchCandidates:
    def __init__(self, cities):
        self.cities = cities   
        
    def __call__(self, paths):
        N = len(self.cities)
        paths = deepcopy(paths)
        start = time.time()
        closest = np.argpartition(self.cities, 11, axis=1)[:,:11]      
        while True:
            best_move, best_delta = None, 0
            for a in range(N):
                for b in closest[a]:
                    if a == b: 
                        continue
                    (cycle1, index1), (cycle2, index2) = find_node(paths, a), find_node(paths, b)
                    move, delta = None, None
                    if cycle1 == cycle2:
                        cycle = paths[cycle1]
                        n = len(cycle)
                        a, b, c, d = a, cycle[(index1+1)%n], b, cycle[(index2+1)%n]
                        delta = delta_swap_edge(self.cities, a, b, c, d)
                        move = delta, EDGE, a, b, c, d
                    else:
                        delta, move = make_swap_node(self.cities, paths, cycle1, index1, cycle2, index2)
                    if delta < best_delta:
                        best_delta, best_move = delta, move                  
            if best_move is None:
                break           

            apply_move(paths, best_move)   
        return time.time() - start, paths
    
class SearchMemory:
    def __init__(self, cities):
        self.cities = cities
    
    def next_moves(self, paths, move):
        kind = move[1]
        moves = []
        if kind == EDGE:
            _, _, a, b, c, d = move
            if a in paths[0]:
                cycle = paths[0] 
            else:
                paths[1]
            n = len(cycle)
            for i, j in gen_swap_edge(n):
                delta, a, b, c, d = gen_swap_edge_2(self.cities, cycle, i, j)
                if delta < 0:
                    moves.append((delta, EDGE, a, b, c, d))
                
        elif kind == NODE:
            _, _, cycle1, cycle2, _, y1, _, _, y2, _ = move
            index1, index2 = paths[cycle1].index(y2), paths[cycle2].index(y1)
            i, j = len(paths[cycle1]), len(paths[cycle2])
            for k in range(j):
                delta, move = make_swap_node(self.cities, paths, cycle1, index1, cycle2, k)
                if delta < 0: moves.append(move)
            for k in range(i):
                delta, move = make_swap_node(self.cities, paths, cycle2, index2, cycle1, k)
                if delta < 0: moves.append(move)

        return moves
    
    def __call__(self, paths):
        paths = deepcopy(paths)
        start = time.time()
        moves = sorted(init_moves(self.cities, paths), key=lambda x: x[0])
        while True:
            to_delete = []
            best_move = None
            for candidate, move in enumerate(moves):
                kind = move[1]
                if kind == EDGE:
                    _, _, a, b, c, d = move
                    (cycle1, s1), (cycle2, s2) = any_has_edge(paths, a, b), any_has_edge(paths, c, d)
                    if cycle1 != cycle2 or s1 == 0 or s2 == 0:
                        to_delete.append(candidate)
                    elif s1 == s2 == +1:
                        to_delete.append(candidate)
                        best_move = move
                        break
                    elif s1 == s2 == -1:
                        to_delete.append(candidate)
                        best_move = move[0], EDGE, b, a, d, c
                        break
                elif kind == NODE:
                    _, _, cycle1, cycle2, x1, y1, z1, x2, y2, z2 = move
                    s1 = has_edge(paths[cycle1], x1, y1)
                    s2 = has_edge(paths[cycle1], y1, z1)
                    s3 = has_edge(paths[cycle2], x2, y2)
                    s4 = has_edge(paths[cycle2], y2, z2)
                    
                    if cycle1 == cycle2 or s1 == 0 or s2 == 0 or s3 == 0 or s4 == 0:
                        to_delete.append(candidate)
                    elif s1 == s2 and s3 == s4:
                        to_delete.append(candidate)
                        best_move = move
                        break                 
            if best_move is None:
                break              
            remove_at(moves, to_delete)
            apply_move(paths, best_move)
            new_moves = self.next_moves(paths, best_move)
            moves = sorted(list(set(moves).union(set(new_moves))), key=lambda x: x[0])
            
        return time.time() - start, paths
    
score_results = []
time_results = []
for file in ['kroa.csv','krob.csv']:
    coords = pd.read_csv(file, sep=' ')
    positions=np.array([coords['x'], coords['y']]).T
    cities = np.round(pairwise_distances(np.array(positions)))
    local_variants = [Steepest(cities), SearchCandidates(cities), SearchMemory(cities)]
    for solve in [regret]:
        solutions = list(map(solve, [(cities, i) for i in range(100)]))
        scores = [score(cities, x) for x in solutions]
        score_results.append(dict(file=file, function=solve.__name__, search="none", variant="none", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
        best_idx = np.argmin(scores)
        best = solutions[best_idx]
        plot_optimized_tours(positions, *best, f'cycle - {solve.__name__}')
        for local_search in local_variants:
            times, new_solutions = zip(*list(map(local_search, solutions)))
            new_scores = [score(cities, x) for x in new_solutions]
            best = new_solutions[best_idx]
            plot_optimized_tours(positions, *best, f'cycle - {solve.__name__}, method - {(type(local_search).__name__).lower()}')
            score_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
            time_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
scores = pd.DataFrame(score_results)
times = pd.DataFrame(time_results)