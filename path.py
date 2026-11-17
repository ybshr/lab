#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import norm
import time

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose
from tools import setcubeplacement

# RRT parameters
K = 1000                 # max number of RRT iterations
DELTA = 0.1              # max distance (m) the cube moves per new node
STEP = 0.01              # cube interpolation step (m) when projecting an edge
GOAL_BIAS = 0.1          # probability of growing straight toward the goal

# Sampling box for random cube positions (m)
X_RANGE = (0.30, 0.50)
Y_RANGE = (-0.40, 0.25)
Z_RANGE = (0.93, 1.30)

# robot and cube are needed by IK but are not arguments of computepath:
# they are set by __main__, or created on first use if this file is imported.
robot, cube = None, None

def _ensure_setup():
    global robot, cube
    if robot is None or cube is None:
        from setup_pinocchio import setuppinocchio
        robot, _, _, cube = setuppinocchio()


# ---------------------------------------------------------------- helpers

def distance(a, b):
    '''Euclidean distance between two vectors.'''
    return norm(b - a)

def lerp(a, b, t):
    '''Linear interpolation between a and b, t in [0, 1].'''
    return a * (1 - t) + b * t

def lerpcube(cA, cB, t):
    '''Cube placement between cA and cB (same rotation, interpolated position).'''
    return pin.SE3(cA.rotation, lerp(cA.translation, cB.translation, t))

def nsteps(cA, cB):
    '''Number of interpolation steps so that each step moves the cube at most STEP.'''
    return max(1, int(np.ceil(distance(cA.translation, cB.translation) / STEP)))


# ---------------------------------------------------------------- II.a sampling

def RAND_CUBE():
    '''Random cube placement inside the sampling box for which a valid grasp exists.
       Returns (q, cube_placement).'''
    while True:
        p = np.array([np.random.uniform(*X_RANGE),
                      np.random.uniform(*Y_RANGE),
                      np.random.uniform(*Z_RANGE)])
        cube_rand = pin.SE3(CUBE_PLACEMENT.rotation, p)
        q, success = computeqgrasppose(robot, robot.q0, cube, cube_rand)
        if success:
            return q, cube_rand


# ---------------------------------------------------------------- tree

# Each node of G is (parent_index, q, cube_placement, segment)
# segment = list of configurations from the parent (excluded) to this node (included)

def NEAREST_VERTEX(G, cube_rand):
    '''Index of the node of G whose cube is closest to cube_rand.'''
    best_i, best_d = None, np.inf
    for i, (_, _, c, _) in enumerate(G):
        d = distance(c.translation, cube_rand.translation)
        if d < best_d:
            best_i, best_d = i, d
    return best_i

def ADD_EDGE_AND_VERTEX(G, parent, q, cube_placement, segment):
    G += [(parent, q, cube_placement, segment)]

def getpath(G):
    '''Walk back from the last node to the root and concatenate the segments.'''
    segments = []
    node = G[-1]
    while node[0] is not None:
        segments.append(node[3])
        node = G[node[0]]
    path = [node[1]]                     # root configuration
    for seg in reversed(segments):
        path += seg
    return path


# ---------------------------------------------------------------- II.b path projection

def projectpath(q_start, cube_start, cube_end):
    '''Carry the cube in a straight line from cube_start to cube_end,
       solving IK at every small step (starting from the previous q).
       Returns (segment, cube_reached, success):
         segment      : list of valid configurations (q_start excluded)
         cube_reached : last cube placement that could be grasped
         success      : True if cube_end was reached'''
    n = nsteps(cube_start, cube_end)
    segment, q_prev, cube_prev = [], q_start, cube_start
    for i in range(1, n + 1):
        c = lerpcube(cube_start, cube_end, i / n)
        q, ok = computeqgrasppose(robot, q_prev, cube, c)   # IK from previous q
        if not ok:                                          # IK failed / collision / limits
            return segment, cube_prev, False
        segment.append(q)
        q_prev, cube_prev = q, c
    return segment, cube_prev, True

def NEW_CONF(q_near, cube_near, cube_rand, delta=DELTA):
    '''Carry the cube from cube_near toward cube_rand (at most delta metres).
       Stops at the last valid step. Returns (segment, cube_new).'''
    cube_end = cube_rand
    d = distance(cube_near.translation, cube_rand.translation)
    if delta is not None and d > delta:
        cube_end = lerpcube(cube_near, cube_rand, delta / d)
    segment, cube_new, _ = projectpath(q_near, cube_near, cube_end)
    return segment, cube_new

def VALID_EDGE(q_near, cube_near, cube_goal):
    '''Can the cube be carried straight from cube_near to cube_goal?
       Returns (success, segment).'''
    segment, _, success = projectpath(q_near, cube_near, cube_goal)
    return success, segment


# ---------------------------------------------------------------- II.c planner

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    _ensure_setup()

    # direct connection first: no tree needed if nothing is in the way
    ok, segment = VALID_EDGE(qinit, cubeplacementq0, cubeplacementqgoal)
    if ok:
        path = [qinit] + segment
    else:
        path = None
        G = [(None, qinit, cubeplacementq0, [])]
        for it in range(K):
            if np.random.rand() < GOAL_BIAS:
                cube_rand = cubeplacementqgoal
            else:
                _, cube_rand = RAND_CUBE()

            i_near = NEAREST_VERTEX(G, cube_rand)
            _, q_near, cube_near, _ = G[i_near]

            segment, cube_new = NEW_CONF(q_near, cube_near, cube_rand)
            if len(segment) == 0:                 # could not move at all
                continue
            ADD_EDGE_AND_VERTEX(G, i_near, segment[-1], cube_new, segment)

            ok, segment = VALID_EDGE(G[-1][1], cube_new, cubeplacementqgoal)
            if ok:
                ADD_EDGE_AND_VERTEX(G, len(G) - 1, segment[-1], cubeplacementqgoal, segment)
                print("Path found after %d iterations (%d nodes)" % (it + 1, len(G)))
                path = getpath(G)
                break

        if path is None:
            print("path not found")
            return [qinit, qgoal]

    # IK may reach the goal cube with a slightly different (equivalent) grasp than qgoal,
    # e.g. another chest angle. Blend toward qgoal in small steps, re-projecting each
    # blended q onto the grasp so the hands never leave the hooks.
    jump = distance(path[-1], qgoal)
    n = int(np.ceil(jump / 0.02))
    qlast = path[-1]
    for i in range(1, n):
        q, ok = computeqgrasppose(robot, lerp(qlast, qgoal, i / n), cube, cubeplacementqgoal)
        if ok:
            path.append(q)
    if jump > 1e-6:
        path.append(qgoal)
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    displaypath(robot,path,dt=0.05,viz=viz)
