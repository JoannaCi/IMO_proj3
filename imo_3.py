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

def random_cycle(args):
    cities, _ = args
    n = cities.shape[0]
    remaining = list(range(n))
    random.shuffle(remaining)
    paths = [remaining[:n//2], remaining[n//2:]]
    return paths

class Greedy(object):
    def __init__(self, cities, variant):
        self.variant = variant
        if variant == "vertices":
            self.delta = delta_replace_vertices_inside
            self.replace = replace_vertices_inside
        if variant == "edges":
            self.delta = delta_replace_edges_inside
            self.replace = replace_edges_inside
        self.cities = cities
        self.moves = [self.outside_vertices_trade_first, self.inside_trade_best]

    def outside_vertices_trade_first(self, cities, paths):
        candidates = outside_candidates(paths)
        random.shuffle(candidates)
        for i, j in candidates:
            score_diff = delta_replace_vertices_outside(cities, paths, i, j)
            if score_diff < 0:
                replace_vertices_outside(paths, i, j)
                return score_diff
        return score_diff

    def inside_trade_best(self, cities, paths):
        path_order = random.sample(range(2), 2)
        for idx in path_order:
            candidates = inside_candidates(paths[idx])
            random.shuffle(candidates)
            for i, j in candidates:
                score_diff = self.delta(cities, paths[idx], i, j)
                if score_diff < 0:
                    self.replace(paths[idx], i, j)
                    return score_diff
        return score_diff
    
    def __call__(self, paths):
        paths = deepcopy(paths)
        start = time.time()
        while True:
            move_order = random.sample(range(2), 2)
            score = self.moves[move_order[0]](self.cities, paths)
            if score >= 0: 
                score = self.moves[move_order[1]](self.cities, paths)
                if score >= 0: 
                    break
        return time.time()-start, paths
    
class Steepest(object):
    def __init__(self, cities, variant):
        self.variant = variant
        self.cities = cities
        if variant == "vertices":
            self.delta_vertices = delta_replace_vertices_inside
            self.replace_vertices = replace_vertices_inside
        elif variant == "edges":
            self.delta_edges = delta_replace_edges_inside
            self.replace_edges = replace_edges_inside
        self.moves = [self.outside_vertices_trade_best, self.inside_trade_best]

    def outside_vertices_trade_best(self, cities, paths):
        candidates = outside_candidates(paths)
        scores = np.array([delta_replace_vertices_outside(cities, paths, i, j) for i, j in candidates])
        best_result_idx = np.argmin(scores)
        if scores[best_result_idx] < 0:
            return replace_vertices_outside, (paths, *candidates[best_result_idx]), scores[best_result_idx]
        return None, None, scores[best_result_idx]

    def inside_trade_best(self, cities, paths):
        combinations_vertices = inside_candidates(paths[0])
        scores_vertices = np.array([self.delta_vertices(cities, paths[0], i, j) for i, j in combinations_vertices])
        best_score_vertices = np.min(scores_vertices)
        
        if self.variant == "edges":
            combinations_edges = inside_candidates(paths[1])
            scores_edges = np.array([self.delta_edges(cities, paths[1], i, j) for i, j in combinations_edges])
            best_score_edges = np.min(scores_edges)
            
            if best_score_edges < best_score_vertices:
                best_combination_idx = np.argmin(scores_edges)
                best_path_idx = 1
                best_score = best_score_edges
            else:
                best_combination_idx = np.argmin(scores_vertices)
                best_path_idx = 0
                best_score = best_score_vertices
        else:
            best_combination_idx = np.argmin(scores_vertices)
            best_path_idx = 0
            best_score = best_score_vertices
            
        if best_score < 0:
            if best_path_idx == 0:
                return self.replace_vertices, (paths[best_path_idx], *combinations_vertices[best_combination_idx]), best_score
            else:
                return self.replace_edges, (paths[best_path_idx], *combinations_edges[best_combination_idx]), best_score
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

    
class RandomSearch(object):
    def __init__(self, cities, time_limit):
        self.cities = cities
        self.time_limit = time_limit
        self.moves = [self.outside_vertices_trade, self.inside_trade_first_vertices, self.inside_trade_first_edges]
        
    def outside_vertices_trade(self, cities, paths):
        candidates = outside_candidates(paths)
        i, j = random.choice(candidates)
        replace_vertices_outside(paths, i, j)

    def inside_trade_first_vertices(self, cities, paths):
        path_idx = random.choice([0,1])
        candidates = inside_candidates(paths[path_idx])
        i, j = random.choice(candidates)
        replace_vertices_inside(paths[path_idx], i, j)
        
    def inside_trade_first_edges(self, cities, paths):
        path_idx = random.choice([0,1])
        candidates = inside_candidates(paths[path_idx])
        i, j = random.choice(candidates)
        replace_edges_inside(paths[path_idx], i, j)
    
    def __call__(self, paths):
        best_solution = paths
        best_score = score(self.cities, paths)
        paths = deepcopy(paths)
        start = time.time()
        while time.time()-start < self.time_limit:
            move = random.choice(self.moves)
            move(self.cities, paths)
            new_score = score(self.cities, paths)
            if new_score<best_score:
                best_solution = deepcopy(paths)
                best_score = new_score
        return best_solution

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

score_final = []
time_final = []
for file in ['kroa.csv','krob.csv']:
    coords = pd.read_csv(file, sep=' ')
    positions=np.array([coords['x'], coords['y']]).T
    cities = np.round(pairwise_distances(np.array(positions)))
    for cycles in [random_cycle, regret]:
        solutions = list(map(cycles, [(cities, i) for i in range(100)]))
        scores = [score(cities, x) for x in solutions]
        score_final.append(dict(file=file, function=cycles.__name__, search="none", variant="none", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
        best = solutions[np.argmin(scores)]
        plot_optimized_tours(positions, *best, f'cycle - {cycles.__name__}')
        for method in [Greedy(cities, "vertices"), Steepest(cities, "vertices"), Greedy(cities, "edges"), Steepest(cities, "edges")]:
            times, new_solutions = zip(*list(map(method, solutions)))
            new_scores = [score(cities, x) for x in new_solutions]
            best = new_solutions[np.argmin(scores)]
            plot_optimized_tours(positions, *best, f'cycle - {cycles.__name__}, neighbourhood - {method.variant}, method - {(type(method).__name__).lower()}')
            score_final.append(dict(file=file, function=cycles.__name__, search=type(method).__name__, variant=method.variant, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
            time_final.append(dict(file=file, function=cycles.__name__, search=type(method).__name__,variant=method.variant, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
        if cycles.__name__ == "random_sol":
            temp_pd = pd.DataFrame(time_final)
            time_limit = max(temp_pd[temp_pd["file"]==file]["mean"])
            random_search = RandomSearch(cities, time_limit)
            random_solutions = list(map(random_search, solutions))
            random_scores = [score(cities, x) for x in random_solutions]
            best = random_solutions[np.argmin(scores)]
            plot_optimized_tours(positions, *best, f'cycle - {cycles.__name__}, method - {(type(random_search).__name__).lower()}')
            score_final.append(dict(file=file, function=cycles.__name__, search=type(random_search).__name__, variant="none", min=int(min(random_scores)), mean=int(np.mean(random_scores)), max=int(max(random_scores))))
scores_final_df = pd.DataFrame(score_final)
times_final_df = pd.DataFrame(time_final)
print(scores_final_df.to_string())
print(times_final_df.to_string())
