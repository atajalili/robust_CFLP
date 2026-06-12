"""
Gurobi modelling for HIV Early Infant Diagnosis VRP
=====================================================================
Supports : ATAT_Min, MTAT_Min, TRL_Min, MRL_Min objectives
Supports : Single and Multiple vehicle modes

Output: Same 13-element list as solver() plus diagnostics dict at index 13.

Usage
-----
result = gurobi_solver(
    "ATAT_Min", "Multiple"
)
solution, lengthes, start_times, allocations, num_vehicles,
    H, TATs, obj_val, sol_time, TRL, MRL, ATAT, MTAT, diagnostics = result
"""

import gurobipy as gp
from gurobipy import GRB
import math
from math import floor, sqrt
from math import sqrt, exp

# =============================================================================
# Gurobi implementation of the model
# =============================================================================

def gurobi_solver(obj,mode, C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,):

    rho   = sum( demand[c] for c in C) / (K*mu)

    m = gp.Model()

    #m.Params.OutputFlag = 0
    #m.Params.FeasibilityTol = 1e-7
    #m.Params.IntFeasTol = 1e-7
    #m.Params.OptimalityTol = 1e-8

    #m.Params.MIPFocus = 1        # focus on finding feasible solutions fast
    #m.Params.Cuts = 2            # aggressive cuts
    #m.Params.Presolve = 2        # aggressive presolve
    #m.Params.Heuristics = 0.3   # more time on heuristics
    #m.Params.TimeLimit = 300     # set a time limit and use best solution found
    
    x = m.addVars(N, N, R, vtype='B')
    y = m.addVars(C, R, vtype='B')

    if mode == "Single":
        z = [1] * Rn
    elif mode == "Multiple": 
        z = m.addVars(R, vtype=GRB.INTEGER, lb=0)  # number of vehicles on route r
        zeta = m.addVars(R,V, vtype='B')           # one if v vehicles on route r
        delta = m.addVars(R,V, vtype='C', lb=0)    # delta[r,v] = zeta[r,v] * H[r]
    
    L = m.addVars(R, vtype='C', lb=0)        # route lengthe
    H = m.addVars(R, vtype='C', lb=0)        # time between succesive visits to a single point on route r
    D = m.addVars(R, vtype='C', lb=0)        # demand collected by each vehicle on route r 
    U = m.addVars(C, R, vtype='C', lb=0)     # transportation time from clinic i on a route r
    q = m.addVars(C, R, vtype='C', lb=0)     # capacity booked on a vehicle on route r for clinic i 
    T = m.addVars(N, R, vtype='C', lb=0)     # time of visiting node i on route r 
    #rho = m.addVar(vtype='C', lb=0, ub=1 - 1e-6)          # lab resource utilization
    
    # Auxiliary variables for perforamnce measures
    TLT     = m.addVars(C, vtype='C', lb=0)   # To Lab Time (TAT minus W_lab)
    max_L   = m.addVar(vtype='C', lb=0)       # max route lengthe
    max_TLT = m.addVar(vtype='C', lb=0)       # Max To Lab Time 
    #W_lab   = m.addVar(vtype='C', lb=0)       # lab delay

    # Auxiliary variables for coninc constraints
    #W = m.addVars(R, vtype='C', lb=0)         # W[r] = H[r] - sum(t[i,j]*omega[i,j,r] for i in N for j in N)
    #S = m.addVar(vtype='C', lb=0)             # S = 1 - rho
    #alpha = m.addVar(vtype='C', lb=0)         # alpha = 1/rho
    beta  = m.addVars(R, vtype='C', lb=0)     # beta[r] = D[r]^2 / W[r]
    #gamma = m.addVars(C, R, vtype='C', lb=0)  # gamma[i,r] = y[i,r]^2 / S 
    
    # Auxiliary variables for linearization
    eta   = m.addVars(C, R, vtype='C', lb=0)              # eta[i,r] = y[i,r] * H[r]  
    #omega = m.addVars(N, N, R, vtype='C', lb=0, ub=1)     # omega[i,j,r] = x[i,j,r] * rho
    theta = m.addVars(C, R, vtype='C')                    # theta[i,r] = y[i,r] * U[i,r]
    #kappa = m.addVars(R,V, vtype='C', lb=0)               # kappa[r,v] = zeta[r,v] * D[r]
    
    #### objective ##################################
    ######## total route length minimization (TRL_Min)
    ######## max route length minimization (MRL_Min)
    ######## average TAT minimization (ATAT_Min)
    ######## maximum TAT minimization (MTAT_Min)

    if obj == "TRL_Min":
        
        m.setObjective(gp.quicksum( L[r] for r in R ), GRB.MINIMIZE )
        
    elif obj == "MRL_Min":
        
        m.setObjective(max_L, GRB.MINIMIZE )
    
    elif obj == "ATAT_Min":

        m.setObjective(gp.quicksum( 0.5 * eta[i,r] + theta[i,r] for i in C for r in R)/Cn
                       + (K-1)/(2*K*mu*rho)
                       + gp.quicksum(beta[r] for r in R)/(2*K*K*mu*mu*(1-rho))
                       + SCV*rho/(2*mu*(1-rho)) + 1/mu , GRB.MINIMIZE )

    elif obj == "MTAT_Min":

        m.setObjective(max_TLT
                       + (K-1)/(2*K*mu*rho)
                       + gp.quicksum(beta[r] for r in R)/(2*K*K*mu*mu*(1-rho))
                       + SCV*rho/(2*mu*(1-rho)) + 1/mu, GRB.MINIMIZE )
    
    ############### constraints

    for i in C:
        m.addConstr( gp.quicksum( x[i,j,r] for j in N for r in R  ) == 1 )                           ## (6)  

        if obj == "MTAT_Min":
            m.addConstr( TLT[i] == gp.quicksum( 0.5 * eta[i,r] + theta[i,r] for r in R) )            ## (49)
            m.addConstr( TLT[i] <= max_TLT)
    
        for r in R:
            m.addConstr( gp.quicksum( x[i,j,r] for j in N ) == gp.quicksum( x[j,i,r] for j in N ) )  ## (9)
    
            m.addConstr( y[i,r] == gp.quicksum( x[i,j,r] for j in N ) )                              ## (1)
    
            m.addConstr( q[i,r] == demand[i] * eta[i,r] + y[i,r] )                                   ## (27)
            #m.addConstr( eta[i,r] <= H[r] )                                                         ## ()
            #m.addConstr( eta[i,r] <= M*y[i,r] )                                                     ## ()
            #m.addConstr( eta[i,r] >= H[r] - M*(1-y[i,r]) )                                          ## (30-32)
            m.addConstr( (y[i,r]==1) >> (eta[i,r] == H[r]) )                                          
            m.addConstr( (y[i,r]==0) >> (eta[i,r] == 0) )                                             

            #m.addQConstr( y[i,r] * y[i,r] <= S * gamma[i,r]  )                                      ## ()
            
            m.addConstr( U[i,r] == L[r] - T[i,r] )                                                   ## (16)
            #m.addConstr( theta[i,r] <= U[i,r] )                                                     ## ()
            #m.addConstr( theta[i,r] <= M*y[i,r] )                                                   ## ()
            #m.addConstr( theta[i,r] >= U[i,r] - M*(1-y[i,r]) )                                      ## (33-35)
            m.addConstr( (y[i,r]==1) >> (theta[i,r] == U[i,r]) )                                         
            m.addConstr( (y[i,r]==0) >> (theta[i,r] == 0) )                                             
    
    for r in R:
        m.addConstr( gp.quicksum( x[0,j,r] for j in N if j != 0 ) == 1 )                            ## (7)
        m.addConstr( gp.quicksum( x[i,Cn+1,r] for i in N if i != (Cn+1) ) == 1 )                    ## (8)
        m.addConstr( L[r] ==  gp.quicksum( t[i,j]*x[i,j,r] for i in N for j in N ) )                ## (2)
        if mode == "Single":
            m.addConstr( L[r] == H[r] )                                                             ## (3) - Single
        elif mode == "Multiple": 
            m.addConstr( z[r] == gp.quicksum( v * zeta[r,v] for v in V ) )                          ## (24) - Multiple
            m.addConstr( gp.quicksum( zeta[r,v] for v in V) <= 1  )                                 ## (25) - Multiple
            m.addConstr( L[r] == gp.quicksum( v * delta[r,v] for v in V) )                          ## (26) - Multiple
            
            for v in V:
                #m.addConstr( delta[r,v] <= H[r] )                                                   ## () - Multiple
                #m.addConstr( delta[r,v] <= M * zeta[r,v] )                                          ## () - Multiple
                #m.addConstr( delta[r,v] >= H[r] - M * ( 1 - zeta[r,v] ) )                           ## (36-38) - Multiple
                m.addConstr( (zeta[r,v]==1) >> (delta[r,v] == H[r]) )                                          
                m.addConstr( (zeta[r,v]==0) >> (delta[r,v] == 0) ) 

                #m.addConstr( (zeta[r,v]==1) >> (kappa[r,v] == D[r]) )                                          
                #m.addConstr( (zeta[r,v]==0) >> (kappa[r,v] == 0) )  
        
        m.addConstr( gp.quicksum( q[i,r] for i in C ) <= Q )                                        ## (11)
    
        m.addConstr( T[0,r] == 0 )                                                                  ## (13)
        m.addConstr( T[Cn+1,r] == L[r] )                                                            ## (14)

        m.addConstr( D[r] == gp.quicksum( demand[i] * eta[i,r] for i in C)  )                       ## (28)

        if obj == "MRL_Min":
            m.addConstr( L[r] <= max_L)                                                             ## Defining L_max
        
        for i in N:
            m.addConstr( x[i,i,r] == 0 )                                                            ## No link to self 
            for j in N:
                #m.addConstr( T[i,r] + t[i,j] - T[j,r] - M*(1-x[i,j,r]) <= 0 )                       ## (29)
                m.addConstr( (x[i,j,r]==1) >> (T[i,r] + t[i,j] - T[j,r] == 0) )

                #m.addConstr( omega[i,j,r] <= rho )                                                  ## ()
                #m.addConstr( omega[i,j,r] <= x[i,j,r] )                                             ## ()
                #m.addConstr( omega[i,j,r] >= rho - ( 1 - x[i,j,r] ) )                               ## ()
                #m.addConstr( (x[i,j,r]==1) >> (omega[i,j,r] == rho) )                                          
                #m.addConstr( (x[i,j,r]==0) >> (omega[i,j,r] == 0) )  
                        
        #m.addConstr( W[r] == L[r] - gp.quicksum(t[i,j]*omega[i,j,r] for i in N for j in N) )        ## ()
        m.addQConstr( D[r] * D[r] <= H[r] * beta[r]   )                                              ## (quadratic constraint) 

    m.addConstr( gp.quicksum( z[r] for r in R) <= Vn )                                               ## (12)
    
    #m.addConstr( rho == (1/(K*mu)) * gp.quicksum( demand[i] * y[i,r] for i in C for r in R ) )      ## ()

    #m.addQConstr( 1 <= rho * alpha )                                                                ## ()

    #m.addConstr( S == 1 - rho  )                                                                    ## ()
    
    #m.addConstr( W_lab == (K-1)*alpha/(2*K*mu)
    #                   + gp.quicksum(beta[r] for r in R)/(2*K*K*mu*mu)
    #                   + SCV*gp.quicksum(demand[i]*gamma[i,r] for i in C for r in R )/(2*K*mu*mu) + 1/mu )  ## Define W_lab

    ### strengthening formulation

    for r in range(Rn - 1):
        m.addConstr(
            gp.quicksum(y[i, r] for i in C) >= gp.quicksum(y[i, r+1] for i in C)
        )

    #############################
    
    m.update()
    m.optimize()
    
    ################################

    ####
    W_lab   = (K-1)/(2*K*mu*rho) + sum(beta[r].x for r in R)/(2*K*K*mu*mu*(1-rho)) + + SCV*rho/(2*mu*(1-rho)) + 1/mu

    W_lab_0 = (K-1)/(2*K*mu*rho) + sum(D[r].x * D[r].x / H[r].x for r in R if H[r].x!= 0)/(2*K*K*mu*mu*(1-rho)) + SCV*rho/(2*mu*(1-rho)) + 1/mu
    
    print(f'Lab delay as in the original version: \t {W_lab_0}')
    print(f'Lab delay as in the reformulated version: \t {W_lab}')
    #########
    solution = {(i,j,r):  x[i,j,r].x for i in N for j in N for r in R if x[i,j,r].x > 0.5}
    
    lengthes = {r: L[r].x for r in R}

    time_between_visits = {r: H[r].x for r in R}
    
    start_times = {(i,r): T[i,r].x for i in N for r in R}
    
    allocations = {(i,r): y[i,r].x for i in C for r in R}

    TATs = {i: sum( 0.5 * eta[i,r].x + theta[i,r].x for r in R) + W_lab for i in C}

    if mode == "Single":
        num_vehicles = {r: 1 for r in R}
    elif mode == "Multiple":
        num_vehicles = {r: z[r].x for r in R}

    objective_value = m.ObjVal
    solution_time = m.runtime
    

    # Calculating all KPIs
    TRL = sum(lengthes.values())
    MRL = max(lengthes.values())
    ATAT = sum(TATs.values())/Cn
    MTAT = max(TATs.values())

    return [solution, lengthes, start_times, allocations, num_vehicles, time_between_visits, TATs, objective_value, solution_time, TRL, MRL, ATAT, MTAT]
