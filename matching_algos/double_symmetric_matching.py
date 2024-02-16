import logging
import time
import numpy as np
import pandas as pd

from find_matchups import TaskInput, TaskOutput
from functools import cache
from matching_algos.base_matching_algo import BaseMatchingAlgo, MatchupCostCalculator
from matching_algos.random_matching import RandomMatcher


class Node:
    def __init__(self, include, exclude):
        self.include = include[:]
        self.exclude = exclude[:]

    def exclude_set(self):
        s = set()
        for i, j in self.exclude:
            s.add(i)
            s.add(j)
        return s

    def include_set(self):
        s = set()
        for i, j in self.include:
            s.add(i)
            s.add(j)
        return s

    def branch(self, node):
        i, j = node
        if i != j:
            nodes = [(i, j), (j, i)]
        else:
            nodes = [(i, i)]
        return Node(self.include + nodes, self.exclude), Node(self.include, self.exclude + nodes)

    def cardinality(self):
        return len(self.include)  # + len(self.exclude) # or just include?

    def __str__(self):
        return str(self.include) + " --- " + str(self.exclude)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(frozenset(self.include)) ^ hash(frozenset(self.exclude))


def remaining_cost_matrix(C, N):
    RCM = C.copy()
    for i, j in N.include:
        RCM = RCM.drop(i, axis=0)
        RCM = RCM.drop(j, axis=1)

    for i, j in N.exclude:
        if i not in N.include_set() and j not in N.include_set():
            RCM.at[i, j] = np.Inf

    #     print("remaining cost matrix:\n", RCM)
    return RCM


def LB(C, N, reduce_rcm):
    # print("Calculating lower bound for", N)
    cost_of_nodes = sum([C.at[i, j] for i, j in N.include])
    reduction_of_remaining_cost_matrix = reduce_rcm(N)[1]

    return cost_of_nodes + reduction_of_remaining_cost_matrix


def find_pivot(N, reduce_rcm, return_all_pivots=False):
    reduced_RCM = reduce_rcm(N)[0]

    possible_pairs = [(i, j) for i in reduced_RCM.index for j in reduced_RCM.columns if i < j
                      and reduced_RCM.at[i, j] == 0
                      and (i, j) not in N.exclude]

    pairs_with_costs = []
    for i, j in possible_pairs:
        tmp_RCM = reduced_RCM.copy()
        tmp_RCM.at[i, j] = np.Inf
        tmp_RCM.at[j, i] = np.Inf

        cost = tmp_RCM.loc[i].min() + tmp_RCM.loc[j].min() + tmp_RCM[i].min() + tmp_RCM[j].min()

        pairs_with_costs.append(((i, j), cost))

    pivots_sorted = sorted(pairs_with_costs, key=lambda t: t[1], reverse=True)

    if return_all_pivots:
        return [p for p, cost in pivots_sorted]

    if len(pivots_sorted) == 0:
        return None

    return pivots_sorted[0][0]


def reduce_matrix(C):
    cost_of_reduction = 0
    C_star = C.copy()

    # Step 1: subtract minimal element of row
    for i in C_star.index:  # range(C.shape[0]):
        row = C_star.loc[i]

        if row.min() == np.inf:
            return C_star, np.inf

        cost_of_reduction += row.min()
        row -= row.min()

    # Step 2: subtract minimal element in column
    for i in C_star.columns:
        col = C_star[i]
        cost_of_reduction += col.min()
        col -= col.min()

    # Step 3: Make symmetric
    C_star = 0.5 * (C_star + C_star.T)

    # Step 4: check if all rows contain zero element

    while not all([C_star.loc[i].min() == 0 for i in C_star.index]):
        for i in C_star.index:
            if C_star.loc[i].min() != 0:
                row = C_star.loc[i]
                col = C_star[i]

                cost_of_reduction += row.min() + col.min()
                row -= row.min()
                col -= col.min()

                C_star = 0.5 * (C_star + C_star.T)

    return C_star, cost_of_reduction


def get_symmetric_matching(C, timeout=120):
    from scipy.optimize import linear_sum_assignment

    def is_solution_symmetric(col_ind):
        return all([
            col_ind[col_ind[i]] == i and col_ind[i] != i
            for i in range(len(col_ind))
        ])

    _, scipy_solution = linear_sum_assignment(C)
    if is_solution_symmetric(scipy_solution):
        return [(idx, j) for idx, j in enumerate(scipy_solution)]

    else:
        return _get_symmetric_matching(C, timeout)[0].include

def _get_symmetric_matching(C, timeout):
    start_time = time.time()

    if type(C) != pd.core.frame.DataFrame:
        C = pd.DataFrame(C)


    nodes_visited = list()
    # nodes contain (LB, len(include), Node)
    terminal_nodes = [(np.inf, 0, Node([], []))]
    nodes_visited.append(terminal_nodes[0])
    N = terminal_nodes.pop(0)[2]

    remaining_cost_matrix_ = cache(lambda n: remaining_cost_matrix(C, n))
    reduce_rcm_ = cache(lambda n: reduce_matrix(remaining_cost_matrix_(n)))
    LB_ = cache(lambda n: LB(C, n, reduce_rcm_))

    while len(N.include) < len(C.index):
        if time.time() - start_time > timeout:
            raise TimeoutError

        pivot = find_pivot(N, reduce_rcm_)

        if pivot is None:
            raise ValueError("No pivot found")


        # print("Current Node:", nodes_visited[-1])
        # print("Branching on:", pivot)

        new_nodes = [(LB_(n), n.cardinality(), n) for n in
                     N.branch(pivot)]  # only allowing one pivot per note!

        # print("New Nodes:", new_nodes)

        terminal_nodes.extend(new_nodes)  # the good nodes are first

        terminal_nodes = sorted(terminal_nodes, key=lambda t: (t[0], -t[1]))

        nodes_visited.append(terminal_nodes[0])
        N = terminal_nodes.pop(0)[2]

    #     print("Number of nodes visited:", len(nodes_visited))
    #     print("\n".join([str(n) for n in nodes_visited]))
    return N, LB_(N)


def final_node_to_col_ind(N):
    ind = [0] * len(N.include)
    for i, j in N.include:
        ind[i] = j
    return ind


class DoubleSymmetricMatcher(BaseMatchingAlgo):
    def __init__(self, exponent=2):
        super().__init__("DoubleSymmetricMatcher")
        self.e = exponent

    def _find_matching(self, task_input: TaskInput, *args, **kwargs) -> TaskOutput:
        try:
            return self._find_matching_orig(task_input, 40)
        except TimeoutError:
            logging.error("DoubleSymmetric Timeout Error")
            return RandomMatcher(runtime=10, optimize_sets=True).find_matching(task_input)

    def _find_matching_orig(self, task_input: TaskInput, timeout=120) -> TaskOutput:
        n = len(task_input.rating_list)
        cost_calc = MatchupCostCalculator.from_taskinput(task_input)

        # match players to teams

        cost_played_together_team = np.array(
            [[cost_calc.played_together_team_duo(i, j)**self.e for i in range(n)] for j in range(n)],
            dtype=float)
        cost_elo_gap_duo = np.array(
            [[cost_calc.elo_gap_duo(i, j)**self.e for i in range(n)] for j in range(n)],
            dtype=float
        )

        heuristic_team_cost = cost_played_together_team + cost_elo_gap_duo

        for i in range(n):
            heuristic_team_cost[i, i] = np.inf

        teams = get_symmetric_matching(heuristic_team_cost, timeout=timeout)
        teams = [t for t in teams if t[0] < t[1]]  # filter out the second entry
        n_teams = len(teams)

        # match teams to matchups

        cost_elo_gap = np.array(
            [[cost_calc.elo_gap([teams[i]+teams[j]])**self.e for i in range(n_teams)] for j in range(n_teams)],
            dtype=float
        )

        cost_team_diff = np.array(
            [[cost_calc.team_diff([teams[i] + teams[j]])**self.e for i in range(n_teams)] for j in range(n_teams)],
            dtype=float
        )

        cost_played_together = np.array(
            [[cost_calc.played_together([teams[i] + teams[j]])**self.e for i in range(n_teams)] for j in range(n_teams)],
            dtype=float
        )

        final_costs = cost_elo_gap + cost_team_diff + cost_played_together
        for i in range(n//2):
            final_costs[i, i] = np.inf

        matchups = get_symmetric_matching(final_costs, timeout=timeout)
        matchups = [m for m in matchups if m[0] < m[1]]

        matchup_matrix = [teams[m1] + teams[m2] for m1, m2 in matchups]

        return TaskOutput(
            input=task_input,
            matchups=self._indices_to_player_ids(matchup_matrix, task_input),
            players_to_pause=[],
            # cost_time=time.time()-start_time
        )
