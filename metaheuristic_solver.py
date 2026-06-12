"""
Simulated Annealing Metaheuristic for HIV Early Infant Diagnosis VRP
=====================================================================
Supports : ATAT_Min, MTAT_Min objectives
Supports : Single and Multiple vehicle modes

Capacity enforcement (two levels):
  Option A — Capacity-aware construction:
             Initial solution and all neighborhood moves check that the
             resulting route is feasible at max vehicles before accepting.
             A route is truly infeasible only if it violates capacity
             even when z_r = max(V).

  Option B — Capacity-aware moves:
             Every neighborhood move checks feasibility using the
             minimum possible headway H_min = L_r / max(V) before
             calling evaluate_solution(). Infeasible moves are reverted.

Output: Same 13-element list as solver() plus diagnostics dict at index 13.

Usage
-----
result = metaheuristic_solver(
    "ATAT_Min", "Multiple",
    C=C, N=N, R=R, t=t,
    demand=demand, Q=Q, V=V,
    K=K, mu=mu, SCV=SCV,
    Cn=Cn, Vn=Vn, Rn=Rn,
    n_iterations=5000, n_restarts=3
)
solution, lengthes, start_times, allocations, num_vehicles,
    H, TATs, obj_val, sol_time, TRL, MRL, ATAT, MTAT, diagnostics = result
"""

import copy
import random
import time
from math import exp

import numpy as np


# =============================================================================
# 1. CAPACITY HELPERS
# =============================================================================

def min_headway(L_r, V):
    """Smallest achievable headway on a route (all vehicles assigned)."""
    v_max = max(V)
    if v_max > 0 and L_r > 0:
        return L_r / v_max
    return 0.0


def cap_used(clinics, H_r, demand):
    """Capacity consumed per vehicle visit given headway H_r."""
    return sum(demand[i] * H_r + 1 for i in clinics)


def feasible_at_max_vehicles(clinics, L_r, demand, Q, V):
    """
    Return True iff the route can satisfy the capacity constraint Q
    when the maximum number of vehicles is assigned.
    An empty or zero-length route is always feasible.
    """
    if not clinics or L_r == 0:
        return True
    H_min = min_headway(L_r, V)
    return cap_used(clinics, H_min, demand) <= Q


# =============================================================================
# 2. VEHICLE ALLOCATION  (objective-aware greedy)
# =============================================================================

def allocate_vehicles(routes, L, demand_on_route, Q, V, Vn, Rn, obj, T_visits=None):
    """
    Determine the vehicle allocation z_r for each route.

    Step 1: Find the minimum z_r that satisfies the capacity constraint
            for each route.  Return (None, None) if any route is
            infeasible even at max vehicles.

    Step 2: Distribute remaining vehicles greedily, scored by the
            improvement in the chosen objective.

    Returns
    -------
    (z, H) : dicts keyed by route index, or (None, None) if infeasible.
    """
    R = list(range(Rn))
    z = {}

    # ---- Step 1: minimum feasible z per route --------------------------------
    for r in R:
        if not demand_on_route[r]:
            z[r] = 0
            continue

        assigned = False
        for v in range(1, max(V) + 1):
            H_r = L[r] / v if L[r] > 0 else 0.0
            if cap_used(demand_on_route[r], H_r, demand_on_route[r]) <= Q:
                z[r] = v
                assigned = True
                break

        if not assigned:
            return None, None   # truly infeasible route

    
    if sum(z.values()) > Vn:
        return None, None  # minimum feasible allocation exceeds fleet size
    
    # ---- Step 2: greedy distribution of remaining vehicles -------------------
    # ---- Step 2: distribute remaining vehicles --------------------------------
    remaining = Vn - sum(z.values())
    
    if obj in ("MTAT_Min", "ATAT_Min") and T_visits is not None and remaining > 0:#if obj == "MTAT_Min" and T_visits is not None and remaining > 0:
        
        #print(f"  [EXHAUSTIVE] obj={obj}, remaining={remaining}, active={[r for r in R if demand_on_route[r]]}")
        # Exhaustive allocation — enumerate all distributions of remaining
        # vehicles across active routes and pick the one minimising MTAT.
        active = [r for r in R if demand_on_route[r]]
    
        def distribute(extra, routes, current, results):
            if not routes:
                results.append(dict(current))
                return
            if len(routes) == 1:
                current[routes[0]] += extra
                results.append(dict(current))
                current[routes[0]] -= extra
                return
            for k in range(extra + 1):
                current[routes[0]] += k
                distribute(extra - k, routes[1:], current, results)
                current[routes[0]] -= k
    
        distributions = []
        distribute(remaining, active, dict(z), distributions)
    
        best_z    = dict(z)
        best_mtat = float('inf')
    
        for z_candidate in distributions:
            obj_val = 0.0
            for r in R:
                if not demand_on_route[r] or z_candidate[r] == 0:
                    continue
                H_r = L[r] / z_candidate[r]
                for i in demand_on_route[r]:
                    tat = 0.5 * H_r + (L[r] - T_visits[r][i])
                    if obj == "MTAT_Min":
                        obj_val = max(obj_val, tat)
                    else:  # ATAT_Min
                        obj_val += tat
            if obj_val < best_mtat:
                best_mtat = obj_val
                best_z    = z_candidate
    
        z = best_z
    
    else:
        # Greedy distribution for all other objectives
        for _ in range(max(0, remaining)):
            best_r     = None
            best_score = -float('inf')
    
            for r in R:
                if not demand_on_route[r]:
                    continue
                if z[r] >= max(V):
                    continue
                H_cur = L[r] / z[r]       if (z[r] > 0 and L[r] > 0) else 0.0
                H_new = L[r] / (z[r] + 1) if L[r] > 0                else 0.0
                score = H_cur - H_new
                if score > best_score:
                    best_score = score
                    best_r     = r
    
            if best_r is None:
                break
            z[best_r] += 1

    # ---- Compute headways ----------------------------------------------------
    H = {}
    for r in R:
        H[r] = (L[r] / z[r]) if (z[r] > 0 and L[r] > 0) else 0.0

    return z, H


# =============================================================================
# 3. SOLUTION EVALUATION
# =============================================================================

def compute_route_length(route, t):
    """Total travel time along a route sequence."""
    return sum(t[route[k], route[k + 1]] for k in range(len(route) - 1))


def compute_visit_times(route, t):
    """Cumulative visit times for every node in the route."""
    T = {route[0]: 0.0}
    for k in range(1, len(route)):
        T[route[k]] = T[route[k - 1]] + t[route[k - 1], route[k]]
    return T


def evaluate_solution(routes, obj, mode, C, N, t, demand, Q, V,
                      K, mu, SCV, Cn, Vn):
    """
    Evaluate a routing solution and return all KPIs.

    Returns None if the solution is infeasible (capacity violation or
    queueing instability rho >= 1).
    """
    Rn = len(routes)
    R  = list(range(Rn))
    C_set = set(C)

    # ---- Route lengths and visit times ---------------------------------------
    L                = {}
    T_visits         = {}
    clinics_on_route = {}
    demand_on_route  = {}

    for r in R:
        route               = routes[r]
        L[r]                = compute_route_length(route, t)
        T_visits[r]         = compute_visit_times(route, t)
        clinics_on_route[r] = [nd for nd in route if nd in C_set]
        demand_on_route[r]  = {i: demand[i] for i in clinics_on_route[r]}

    # ---- Vehicle allocation --------------------------------------------------
    if mode == "Single":
        z = {r: (1 if clinics_on_route[r] else 0) for r in R}
        H = {r: (L[r] if z[r] > 0 else 0.0)       for r in R}

        # Capacity check for single vehicle
        for r in R:
            if not clinics_on_route[r]:
                continue
            if cap_used(demand_on_route[r], H[r], demand_on_route[r]) > Q:
                return None

    elif mode == "Multiple":
        z, H = allocate_vehicles(
            routes, L, demand_on_route, Q, V, Vn, Rn, obj,
            T_visits=T_visits
        )
        if z is None:
            return None
        for r in R:
            if not clinics_on_route[r]:
                z[r] = 0
                H[r] = 0.0

    # ---- Lab utilisation and stability ---------------------------------------
    rho = (1.0 / (K * mu)) * sum(
        demand[i] for r in R for i in clinics_on_route[r]
    )
    if rho >= 1.0:
        return None

    # ---- Auxiliary variables -------------------------------------------------
    D     = {r: sum(demand[i] * H[r] for i in clinics_on_route[r]) for r in R}
    U     = {(i, r): L[r] - T_visits[r][i]
             for r in R for i in clinics_on_route[r]}
    eta   = {(i, r): (H[r] if i in clinics_on_route[r] else 0.0)
             for r in R for i in C}
    theta = {(i, r): (U[i, r] if i in clinics_on_route[r] else 0.0)
             for r in R for i in C}

    # ---- Lab delay (Equation 21) ---------------------------------------------
    sum_D2_H = sum(D[r] ** 2 / H[r] for r in R if H[r] > 0)

    W_lab = (
        (K - 1) / (2 * K * mu * rho)
        + sum_D2_H / (2 * K * K * mu * mu * (1 - rho))
        + SCV * rho / (2 * mu * (1 - rho))
        + 1.0 / mu
    )

    # ---- Clinic-level TAT ----------------------------------------------------
    TATs = {
        i: sum(0.5 * eta[i, r] + theta[i, r] for r in R) + W_lab
        for i in C
    }

    # ---- Aggregate KPIs ------------------------------------------------------
    TRL  = sum(L.values())
    MRL  = max(L.values()) if L else 0.0
    ATAT = sum(TATs.values()) / len(C)
    MTAT = max(TATs.values())

    obj_val = {"ATAT_Min": ATAT,
               "MTAT_Min": MTAT,
               "TRL_Min":  TRL,
               "MRL_Min":  MRL}.get(obj, ATAT)

    # ---- Build output mirrors of solver() format -----------------------------
    solution = {
        (routes[r][k], routes[r][k + 1], r): 1
        for r in R for k in range(len(routes[r]) - 1)
    }
    start_times = {(i, r): T_visits[r].get(i, 0.0) for r in R for i in N}
    allocations = {(i, r): (1.0 if i in clinics_on_route[r] else 0.0)
                   for i in C for r in R}

    return {
        "solution":            solution,
        "lengthes":            L,
        "start_times":         start_times,
        "allocations":         allocations,
        "num_vehicles":        z,
        "time_between_visits": H,
        "TATs":                TATs,
        "objective_value":     obj_val,
        "TRL":   TRL,
        "MRL":   MRL,
        "ATAT":  ATAT,
        "MTAT":  MTAT,
        "rho":   rho,
        "W_lab": W_lab,
        "routes": routes,
    }


# =============================================================================
# 4. INITIAL SOLUTION  (Option A — capacity-aware nearest-neighbour)
# =============================================================================

def generate_initial_solution(C, Rn, Cn, t, demand, Q, V, max_attempts=500):
    """
    Build an initial solution using a greedy nearest-neighbour heuristic.

    Option A: After constructing each route, check that it is feasible
    at max vehicles.  If not, try a new random shuffle.  Falls back to
    one-clinic-per-route after max_attempts.
    """
    C_list = list(C)

    for _ in range(max_attempts):
        random.shuffle(C_list)

        # Round-robin assignment
        route_clinics = [[] for _ in range(Rn)]
        for idx, clinic in enumerate(C_list):
            route_clinics[idx % Rn].append(clinic)

        # Nearest-neighbour ordering within each route
        routes = []
        for r in range(Rn):
            if not route_clinics[r]:
                routes.append([0, Cn + 1])
                continue
            unvisited = route_clinics[r][:]
            route = [0]
            while unvisited:
                nearest = min(unvisited, key=lambda j: t[route[-1], j])
                route.append(nearest)
                unvisited.remove(nearest)
            route.append(Cn + 1)
            routes.append(route)

        # Option A: check every route is feasible at max vehicles
        L = {r: compute_route_length(routes[r], t) for r in range(Rn)}
        clinics_on_route = {
            r: [nd for nd in routes[r] if nd in set(C)]
            for r in range(Rn)
        }

        if all(
            feasible_at_max_vehicles(clinics_on_route[r], L[r], demand, Q, V)
            for r in range(Rn)
        ):
            return routes

    # Fallback: spread clinics as thinly as possible
    random.shuffle(C_list)
    routes = [[0, Cn + 1] for _ in range(Rn)]
    for idx, clinic in enumerate(C_list):
        r = idx % Rn
        routes[r].insert(-1, clinic)
    return routes


# =============================================================================
# 5. NEIGHBORHOOD MOVES  (Option B — capacity-aware)
# =============================================================================

def _clinics_in(route, C_set):
    return [nd for nd in route if nd in C_set]


def _route_ok(route, C_set, t, demand, Q, V):
    """Quick feasibility check for a candidate route (Option B)."""
    clinics = _clinics_in(route, C_set)
    L_r = compute_route_length(route, t)
    return feasible_at_max_vehicles(clinics, L_r, demand, Q, V)


def move_clinic(routes, C_set, Cn, t, demand, Q, V):
    """Relocate a random clinic to a random position in a random route."""
    new_routes = copy.deepcopy(routes)
    Rn = len(new_routes)

    non_empty = [r for r in range(Rn) if _clinics_in(new_routes[r], C_set)]
    if not non_empty:
        return new_routes

    r_from  = random.choice(non_empty)
    clinics = _clinics_in(new_routes[r_from], C_set)
    clinic  = random.choice(clinics)
    new_routes[r_from].remove(clinic)

    r_to       = random.randint(0, Rn - 1)
    insert_pos = random.randint(1, len(new_routes[r_to]) - 1)
    new_routes[r_to].insert(insert_pos, clinic)

    # Option B: revert if capacity-infeasible
    if not _route_ok(new_routes[r_to], C_set, t, demand, Q, V):
        return copy.deepcopy(routes)

    return new_routes


def swap_clinics(routes, C_set, Cn, t, demand, Q, V):
    """Swap a clinic from r1 with a clinic from r2."""
    new_routes = copy.deepcopy(routes)
    Rn = len(new_routes)

    non_empty = [r for r in range(Rn) if _clinics_in(new_routes[r], C_set)]
    if len(non_empty) < 2:
        return new_routes

    r1, r2  = random.sample(non_empty, 2)
    c1_list = _clinics_in(new_routes[r1], C_set)
    c2_list = _clinics_in(new_routes[r2], C_set)
    if not c1_list or not c2_list:
        return new_routes

    c1 = random.choice(c1_list)
    c2 = random.choice(c2_list)

    i1 = new_routes[r1].index(c1)
    i2 = new_routes[r2].index(c2)
    new_routes[r1][i1] = c2
    new_routes[r2][i2] = c1

    # Option B: revert if either route becomes infeasible
    if (not _route_ok(new_routes[r1], C_set, t, demand, Q, V) or
            not _route_ok(new_routes[r2], C_set, t, demand, Q, V)):
        return copy.deepcopy(routes)

    return new_routes


def two_opt(routes, C_set, Cn, t, demand, Q, V):
    """Reverse a random segment within a route (2-opt)."""
    new_routes = copy.deepcopy(routes)
    Rn = len(new_routes)

    eligible = [r for r in range(Rn)
                if len(_clinics_in(new_routes[r], C_set)) >= 2]
    if not eligible:
        return new_routes

    r     = random.choice(eligible)
    route = new_routes[r]
    n     = len(route)
    if n <= 3:
        return new_routes

    i = random.randint(1, n - 3)
    j = random.randint(i + 1, n - 2)
    new_routes[r] = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    # 2-opt keeps same clinics on route => capacity unchanged, no check needed
    return new_routes


def or_opt(routes, C_set, Cn, t, demand, Q, V):
    """Move two consecutive clinics as a block to another route/position."""
    new_routes = copy.deepcopy(routes)
    Rn = len(new_routes)

    eligible = [r for r in range(Rn)
                if len(_clinics_in(new_routes[r], C_set)) >= 2]
    if not eligible:
        return new_routes

    r_from = random.choice(eligible)
    route  = new_routes[r_from]
    c_idx  = [k for k in range(len(route)) if route[k] in C_set]
    if len(c_idx) < 2:
        return new_routes

    start   = random.randint(0, len(c_idx) - 2)
    pos1    = c_idx[start]
    pos2    = c_idx[start + 1]
    segment = [route[pos1], route[pos2]]

    new_routes[r_from].pop(pos2)
    new_routes[r_from].pop(pos1)

    r_to       = random.randint(0, Rn - 1)
    insert_pos = random.randint(1, len(new_routes[r_to]) - 1)
    for nd in reversed(segment):
        new_routes[r_to].insert(insert_pos, nd)

    # Option B: revert if destination route becomes infeasible
    if not _route_ok(new_routes[r_to], C_set, t, demand, Q, V):
        return copy.deepcopy(routes)

    return new_routes


def move_worst_clinic(routes, C_set, Cn, TATs, obj, mode,
                      t, demand, Q, V, K, mu, SCV, N, Vn, C):
    """
    Best-insertion move for MTAT_Min.

    Identify the clinic with the highest TAT, enumerate all feasible
    insertion positions across all routes, and select the one that
    minimises the objective.  Option B capacity check is applied before
    each full evaluation.
    """
    Rn           = len(routes)
    worst_clinic = max(TATs, key=TATs.get)

    r_from = next(
        (r for r in range(Rn) if worst_clinic in routes[r]), None
    )
    if r_from is None:
        return copy.deepcopy(routes)

    base = copy.deepcopy(routes)
    base[r_from].remove(worst_clinic)

    best_routes = None
    best_obj    = float('inf')

    for r_to in range(Rn):
        for pos in range(1, len(base[r_to])):
            candidate = copy.deepcopy(base)
            candidate[r_to].insert(pos, worst_clinic)

            # Option B: skip if infeasible at max vehicles
            if not _route_ok(candidate[r_to], C_set, t, demand, Q, V):
                continue

            result = evaluate_solution(
                candidate, obj, mode, C, N, t,
                demand, Q, V, K, mu, SCV, Cn, Vn
            )
            if result is None:
                continue
            if result["objective_value"] < best_obj:
                best_obj    = result["objective_value"]
                best_routes = candidate

    return best_routes if best_routes is not None else copy.deepcopy(routes)


def get_neighbor(routes, C_set, Cn, t, demand, Q, V,
                 obj, mode, N, K, mu, SCV, Vn, C,
                 current_TATs=None):
    """
    Select and apply a neighborhood move.

    Move probabilities for MTAT_Min:
        move=0.21, swap=0.21, two_opt=0.12, or_opt=0.06, worst=0.40
    Move probabilities for all other objectives:
        move=0.35, swap=0.35, two_opt=0.20, or_opt=0.10
    """
    if obj == "MTAT_Min" and current_TATs is not None:
        moves   = ["move", "swap", "two_opt", "or_opt", "worst"]
        weights = [0.21,   0.21,   0.12,      0.06,     0.40]
    else:
        moves   = ["move", "swap", "two_opt", "or_opt"]
        weights = [0.35,   0.35,   0.20,      0.10]

    move = random.choices(moves, weights=weights)[0]

    if move == "move":
        return move_clinic(routes, C_set, Cn, t, demand, Q, V), move
    elif move == "swap":
        return swap_clinics(routes, C_set, Cn, t, demand, Q, V), move
    elif move == "two_opt":
        return two_opt(routes, C_set, Cn, t, demand, Q, V), move
    elif move == "or_opt":
        return or_opt(routes, C_set, Cn, t, demand, Q, V), move
    else:
        return move_worst_clinic(
            routes, C_set, Cn, current_TATs, obj, mode,
            t, demand, Q, V, K, mu, SCV, N, Vn, C
        ), "worst"


# =============================================================================
# 6. SIMULATED ANNEALING
# =============================================================================

def simulated_annealing(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    n_iterations=5000,
    initial_temp=None,
    cooling_rate=0.995,
    n_restarts=3,
    verbose=True,
    time_limit=300,
    no_improve_limit=500,
):
    """
    Simulated Annealing for the HIV EID VRP.

    Temperature is auto-calibrated so that a 5% worse solution is
    accepted with probability 0.8 at the start.

    Early stopping: restart terminates if no improvement for
    no_improve_limit consecutive iterations.
    """
    start_time = time.time()
    C_set      = set(C)

    global_best_routes = None
    global_best_result = None
    global_best_obj    = float('inf')

    convergence_history    = []
    all_iterations_history = []
    restart_history        = []
    n_improving = n_accepted = n_total = 0
    best_found_at_iter = 0
    global_iter        = 0

    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            if verbose:
                print("  Time limit reached.")
            break

        if verbose:
            print(f"\n--- Restart {restart + 1}/{n_restarts} ---")

        restart_history.append(global_iter)

        # ---- Generate feasible initial solution ------------------------------
        current_routes = None
        current_result = None
        for _ in range(200):
            candidate_routes = generate_initial_solution(
                C, Rn, Cn, t, demand, Q, V
            )
            candidate_result = evaluate_solution(
                candidate_routes, obj, mode, C, N, t,
                demand, Q, V, K, mu, SCV, Cn, Vn
            )
            if candidate_result is not None:
                current_routes = candidate_routes
                current_result = candidate_result
                break

        if current_result is None:
            if verbose:
                print("  Could not generate feasible initial solution. Skipping.")
            continue

        current_obj = current_result["objective_value"]

        # Seed global best immediately
        if current_obj < global_best_obj:
            global_best_routes = copy.deepcopy(current_routes)
            global_best_result = current_result
            global_best_obj    = current_obj
            best_found_at_iter = global_iter
            convergence_history.append((global_iter, global_best_obj))

        local_best_routes = copy.deepcopy(current_routes)
        local_best_result = current_result
        local_best_obj    = current_obj

        # ---- Temperature calibration -----------------------------------------
        # Calibrate T on the routing-variable part of the objective only.
        # The constant lab-delay terms (K-1)/(2Kμρ), SCV·ρ/(2μ(1-ρ)), 1/μ
        # dominate when K is large / rho is small, inflating T and turning
        # SA into a random walk.  Subtracting them gives the true search space.
        if initial_temp is not None:
            T = initial_temp
        else:
            _rho = current_result["rho"]
            _const_lab = ((K - 1) / (2 * K * mu * _rho)
                          + SCV * _rho / (2 * mu * (1 - _rho))
                          + 1.0 / mu)
            _routing = abs(current_obj - _const_lab)
            T = (0.05 * _routing / (-np.log(0.8))
                 if _routing > 1e-9
                 else 0.05 * abs(current_obj) / (-np.log(0.8))
                 if current_obj != 0 else 1.0)

        if verbose:
            print(f"  Initial obj: {current_obj:.4f}, T0: {T:.4f}")

        no_improve = 0

        # ---- Main SA loop ----------------------------------------------------
        for iteration in range(n_iterations):
            global_iter += 1

            if time.time() - start_time > time_limit:
                break

            new_routes, move_type = get_neighbor(
                current_routes, C_set, Cn, t, demand, Q, V,
                obj, mode, N, K, mu, SCV, Vn, C,
                current_TATs=current_result["TATs"]
            )

            new_result = evaluate_solution(
                new_routes, obj, mode, C, N, t,
                demand, Q, V, K, mu, SCV, Cn, Vn
            )
            n_total += 1

            if new_result is None:
                T *= cooling_rate
                continue

            new_obj = new_result["objective_value"]
            delta   = new_obj - current_obj

            accepted = False
            if delta < 0:
                accepted = True
                n_improving += 1
            elif T > 1e-10 and random.random() < exp(-delta / T):
                accepted = True

            if accepted:
                n_accepted    += 1
                current_routes = new_routes
                current_result = new_result
                current_obj    = new_obj

                if current_obj < local_best_obj:
                    local_best_routes = copy.deepcopy(current_routes)
                    local_best_result = current_result
                    local_best_obj    = current_obj
                    no_improve        = 0

                    if local_best_obj < global_best_obj:
                        global_best_routes = copy.deepcopy(local_best_routes)
                        global_best_result = local_best_result
                        global_best_obj    = local_best_obj
                        best_found_at_iter = global_iter
                        convergence_history.append(
                            (global_iter, global_best_obj)
                        )
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T *= cooling_rate

            all_iterations_history.append(
                (restart, iteration, current_obj, local_best_obj, T)
            )

            if no_improve >= no_improve_limit:
                if verbose:
                    print(f"  Early stop at iteration {iteration}")
                break

            if verbose and iteration % 500 == 0:
                print(f"  Iter {iteration:5d} | "
                      f"current: {current_obj:.4f} | "
                      f"local best: {local_best_obj:.4f} | "
                      f"global best: {global_best_obj:.4f} | "
                      f"T: {T:.6f}")

    # ---- End of all restarts -------------------------------------------------
    solution_time = time.time() - start_time

    if global_best_result is None:
        raise RuntimeError(
            "Metaheuristic failed to find any feasible solution.\n"
            "Possible causes:\n"
            "  1. rho >= 1  (increase K or mu, or check demand values)\n"
            "  2. Q too tight (increase Q or reduce clinic demands)\n"
            "  3. Instance parameters inconsistent"
        )

    diagnostics = {
        "convergence_history":     convergence_history,
        "all_iterations_history":  all_iterations_history,
        "restart_history":         restart_history,
        "n_iterations_total":      global_iter,
        "n_improving_moves":       n_improving,
        "n_accepted_moves":        n_accepted,
        "n_total_moves":           n_total,
        "acceptance_rate":         n_accepted  / n_total if n_total > 0 else 0,
        "improvement_rate":        n_improving / n_total if n_total > 0 else 0,
        "best_found_at_iteration": best_found_at_iter,
        "n_restarts":              n_restarts,
        "solution_time":           solution_time,
    }

    r = global_best_result

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  FINAL RESULT  ({obj}, {mode})")
        print(f"  Objective : {r['objective_value']:.4f}")
        print(f"  ATAT      : {r['ATAT']:.4f}")
        print(f"  MTAT      : {r['MTAT']:.4f}")
        print(f"  TRL       : {r['TRL']:.4f}")
        print(f"  MRL       : {r['MRL']:.4f}")
        print(f"  Rho       : {r['rho']:.4f}")
        print(f"  W_lab     : {r['W_lab']:.4f}")
        print(f"  Time      : {solution_time:.2f}s")
        print(f"  Best iter : {best_found_at_iter}")
        print(f"  Accept %  : {diagnostics['acceptance_rate']:.1%}")
        print(f"{'=' * 55}")

    return [
        r["solution"],           # 0
        r["lengthes"],           # 1
        r["start_times"],        # 2
        r["allocations"],        # 3
        r["num_vehicles"],       # 4
        r["time_between_visits"],# 5
        r["TATs"],               # 6
        r["objective_value"],    # 7
        solution_time,           # 8
        r["TRL"],                # 9
        r["MRL"],                # 10
        r["ATAT"],               # 11
        r["MTAT"],               # 12
        diagnostics,             # 13
    ]


# =============================================================================
# 7. PUBLIC WRAPPER  — drop-in replacement for solver()
# =============================================================================

def metaheuristic_solver(obj, mode,
                          C, N, R, t, demand, Q, V,
                          K, mu, SCV, Cn, Vn, Rn,
                          n_iterations=5000,
                          n_restarts=3,
                          cooling_rate=0.995,
                          time_limit=300,
                          no_improve_limit=500,
                          verbose=True):
    """
    Drop-in replacement for solver(obj, mode).

    All instance variables must be passed explicitly — they are the
    same globals produced by instance_generator().
    """
    return simulated_annealing(
        obj=obj, mode=mode,
        C=C, N=N, R=R, t=t,
        demand=demand, Q=Q, V=V,
        K=K, mu=mu, SCV=SCV,
        Cn=Cn, Vn=Vn, Rn=Rn,
        n_iterations=n_iterations,
        n_restarts=n_restarts,
        cooling_rate=cooling_rate,
        time_limit=time_limit,
        no_improve_limit=no_improve_limit,
        verbose=verbose,
    )
