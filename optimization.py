from operator import itemgetter

import pandas as pd
import time
import random

import networkx as nx
import re
from collections import defaultdict
from itertools import permutations
from z3 import *
from tqdm.notebook import tqdm as tqdm_notebook


def groups_expressions(exps, is_hashing):
    """ Return groups of identical expressions."""
    groups = defaultdict(list)
    if not is_hashing:
        groups = {(i, j): [(i, j)] for (i, j), e in exps.items()}
    else:
        for (i, j), e in exps.items():
            groups[e.z3_expression].append((i, j))
        groups = {list(pairs)[0]: list(pairs) for k, pairs in groups.items()}
    return groups


def pairs_to_check(encoder, id2expressions, group_representative, is_atom_implication):
    """ Return pairs of expressions to  use a SAT solver to test the satisfiability."""
    if is_atom_implication:
        g = get_atom_implication(encoder, id2expressions, group_representative)
        pairs = set()
        # nodes
        for _, d in g.nodes(data=True):
            for i, j in permutations(d['idx'], 2):
                if i != j:
                    pairs.add((i, j))
        # edged
        for x, y in g.edges():
            for i in g.nodes[x]['idx']:
                # bi=nodes[i]['z3_expression']
                for j in g.nodes[y]['idx']:
                    if i != j:
                        pairs.add((i, j))
        return pairs
    else:
        return permutations(group_representative, 2)


def get_atom_implication(encoder, id2expressions, nodes):
    """ Return pairs of expressions to  use a SAT solver to test the satisfiability using atom base approach."""
    atoms = defaultdict(set)

    for d in nodes:
        for a in [a.strip() for a in re.split(r'And|,|\(|\)|Or', id2expressions[d].z3_expression) if a]:
            atoms[a].add(d)
    g = nx.DiGraph()
    g.add_nodes_from(atoms)
    for atom in atoms:
        g.nodes[atom]['idx'] = atoms[atom]

    all_combinations = permutations(atoms.keys(), 2)
    bool_features = ([f'y_{f}' for f in encoder.features
                      if f in encoder.columns and encoder.encoder[f].size == 2])
    for x, y in all_combinations:
        fx, sx, vx = x.split()
        fy, sy, vy = y.split()

        if fx != fy:
            continue
        elif sx == '==' and sy == '!=' and vx != vy:
            g.add_edge(x, y)
        elif sx == '==' and sy == '>' and vx > vy:
            g.add_edge(x, y)
        elif sx == '==' and sy == '>=' and vx >= vy:
            g.add_edge(x, y)
        elif sx == '==' and sy == '<' and vx < vy:
            g.add_edge(x, y)
        elif sx == '==' and sy == '<=' and vx <= vy:
            g.add_edge(x, y)
        elif sx == '>' and sy == '>' and vx > vy:
            g.add_edge(x, y)
        elif sx == '>' and sy == '>=' and vx > vy:
            g.add_edge(x, y)
        elif sx == '>' and sy == '!=' and vx >= vy:
            g.add_edge(x, y)
        elif sx == '>=' and sy == '>' and vx > vy:
            g.add_edge(x, y)
        elif sx == '>=' and sy == '>=' and vx >= vy:
            g.add_edge(x, y)
        elif sx == '>=' and sy == '!=' and vx > vy:
            g.add_edge(x, y)
        elif sx == '<' and sy == '<' and vx < vy:
            g.add_edge(x, y)
        elif sx == '<' and sy == '<=' and vx <= vy:
            g.add_edge(x, y)
        elif sx == '<' and sy == '!=' and vx <= vy:
            g.add_edge(x, y)
        elif sx == '<=' and sy == '<' and vx < vy:
            g.add_edge(x, y)
        elif sx == '<=' and sy == '<=' and vx <= vy:
            g.add_edge(x, y)
        elif sx == '<=' and sy == '!=' and vx < vy:
            g.add_edge(x, y)
        elif sx == '!=' and sy == '==' and fx in bool_features:
            g.add_edge(x, y)
    return g


def my_solve(u, v, encoder):
    s = Solver()
    # init by encoder
    for i, feature in enumerate(encoder.features):

        if i < len(encoder.columns):  # categorical features
            exec(f'y_{feature}=Int("y_{feature}")')
            exec(f's.add(y_{feature} >= 0)')
            exec(f's.add(y_{feature} <= encoder.cat_vars_ord[i] - 1)')
        else:  # numeric features
            if encoder.is_integer(feature):  # int feature
                exec(f'y_{feature}=Int("y_{feature}")')
            else:  # Real feature
                exec(f'y_{feature}=Real("y_{feature}")')

    u, v = eval(u), eval(v)
    s.add(And(u, Not(v)))
    r = s.check()
    if r == unsat:
        return "no solution"
    elif r == unknown:
        return "failed to solve"
        try:
            print(s.model())
        except Z3Exception:
            return
    else:
        return s.model()


def is_implies(u, v, encoder):
    return my_solve(u, v, encoder) == 'no solution'


def get_implication_graph(encoder, expressions, is_atom_implication=False, is_hashing=False):
    start_time = time.time()
    id2expressions = {(e.view_id, e.prediction_id): e for e in expressions}
    constraint_expressions = {(i, j): e for (i, j), e in id2expressions.items() if e.z3_expression != 'True'}
    no_constraint_expressions = {(i, j): e for (i, j), e in id2expressions.items() if e.z3_expression == 'True'}
    groups = groups_expressions(constraint_expressions, is_hashing)
    groups_representative = groups.keys()
    pairs = pairs_to_check(encoder, id2expressions, groups_representative, is_atom_implication)
    pairs = list(pairs)
    g = nx.DiGraph()
    g.add_nodes_from([((i, j), id2expressions[(i, j)]._asdict()) for (i, j) in groups_representative])

    edges = set()

    for (i, j) in tqdm_notebook(pairs, leave=False, total=len(pairs)):
        if is_implies(id2expressions[i].z3_expression, id2expressions[j].z3_expression, encoder):
            g.add_edge(i, j)
            edges.add((i, j))

    con = nx.condensation(g)
    # update strongly connected components
    if is_hashing:
        for node in con.nodes():
            new_members = set()
            for m in con.nodes[node]['members']:
                new_members.update(groups[m])
            con.nodes[node]['members'] = new_members

    # add no_constraint components
    n = con.number_of_nodes()
    con.add_node(n)
    con.nodes[n]['members'] = set(no_constraint_expressions.keys())
    # add edges from each components to no_constraint components
    con.add_edges_from([(i, n) for i in range(n)])

    # transitive reduction
    tr = nx.transitive_reduction(con)
    tr.graph['id2expressions'] = id2expressions
    tr.add_nodes_from(con.nodes(data=True))
    tr.add_edges_from((u, v, con.edges[u, v]) for u, v in tr.edges)
    tr.graph['GraphGenerationTime'] = time.time() - start_time
    tr.graph['#SAT Calls'] = len(pairs)
    return tr


def get_instance_id(g, node):
    return g.graph['id2expressions'][node].instance_id


def get_constraints(g, node):
    return g.graph['id2expressions'][node].constraints


def get_instance(querylang, g, node):
    return querylang.instances.get_instance(get_instance_id(g, node))


def reconstruct_cfs(instance, changes):
    if changes is None:
        return None
    n_cfs = changes['CfId'].nunique()
    all_values = pd.concat([instance] * n_cfs, ignore_index=True)
    for _, cf_id, f_name, new_val in changes.itertuples():
        all_values.loc[cf_id, f_name] = new_val
    return all_values


def add_cfs(querylang, cf_alg, g, equivalence_class, node):
    """ Generate new CFs.
    Args:
      querylang: QueryLang object.
      cf_alg: CF generator.
      g: implication graph.
      equivalence_class: the equivalence class.
      node: the current node.
    """
    start_time = time.time()
    changes = querylang.get_changed_values(cf_alg, get_instance_id(g, node), get_constraints(g, node))
    g.graph['CFs_times'].append(time.time() - start_time)
    cf = reconstruct_cfs(get_instance(querylang, g, node), changes)
    if cf is not None:
        i = len(g.graph['CFs'])
        g.nodes[equivalence_class]["CFs_ids"].add(i)
        g.graph['CFs'][i] = cf
        g.graph['CFs_Success'][node] = i
    else:
        g.graph['CFs_Success'][node] = None


def l0_metric(x, y):
    x = x.to_numpy().reshape(-1)
    y = y.to_numpy().reshape(-1)
    return sum(x!=y)


def get_best_cfs(querylang, g, equivalence_class, node):
    """ Computed the best CF for node .
     Args:
       querylang: QueryLang object.
       g: implication graph.
       equivalence_class: the equivalence class.
       node: the current node.

    :return the best cf and its distance from the node' instance max_CF_to_check
     """
    start_time = time.time()
    f = lambda i: (i, g.graph['metric'](get_instance(querylang, g, node), g.graph['CFs'][i]))
    to_check = g.nodes[equivalence_class]['CFs_ids'].difference(g.nodes[equivalence_class]["CFs_ids_check"][node])
    if len(to_check) > g.graph['max_CF_to_check']:
        to_check = set(random.sample(to_check, g.graph['max_CF_to_check']))
    node_cfs_success = g.graph['CFs_Success'].get(node)
    if node_cfs_success is not None:
        to_check.add(node_cfs_success)
    best_cf, diff = min(map(f, to_check), key=itemgetter(1))
    g.nodes[equivalence_class]["CFs_ids_check"][node] = to_check.difference(set({best_cf}))
    g.nodes[equivalence_class]['CFs_compare_times'][node].append(time.time() - start_time)
    return best_cf, diff


def batch_cfs_generation(g, d, t, querylang, cf_alg, max_CF_to_check=None):
    """ Assign CF to each node in the (condensation of) implication graph a CF .
     Args:
       g: implication graph.
       d: distance function between two instances.
       t: re-use threshold.
       querylang: QueryLang object.
       cf_alg: CF generator.
       equivalence_class: the equivalence class.
       node: the current node.
       max_CF_to_check (int or None): Maximum number of CFs to check for each prediction.
            If set to None, all possible CFs will be considered, which may be computationally expensive.

    :return Implication graph with CF data
     """
    start_time = time.time()
    g = g.copy()
    g.graph['copy_time'] = time.time() - start_time
    g.graph['CFs'] = {}
    g.graph['CFs_times'] = []
    g.graph['CFs_Success'] = {}
    g.graph['Ans'] = {}
    g.graph['metric'] = d
    g.graph['threshold'] = t
    g.graph['total_CFs_generation_time'] = 0
    g.graph['max_CF_to_check'] = len(g.graph['id2expressions']) if  max_CF_to_check is None else max_CF_to_check
    for equivalence_class in g.nodes():
        g.nodes[equivalence_class]["CFs_ids"] = set()
        g.nodes[equivalence_class]["CFs_ids_check"] = {}
        g.nodes[equivalence_class]['CFs_compare_times'] = {}
        for (i, j) in g.nodes[equivalence_class]['members']:
            g.nodes[equivalence_class]["CFs_ids_check"][(i, j)] = set()
            g.nodes[equivalence_class]['CFs_compare_times'][(i,j)] = []

    for equivalence_class in tqdm_notebook(nx.topological_sort(g), total=len(g.nodes), leave=False):
        for (i, j) in g.nodes[equivalence_class]['members']:
            if len(g.nodes[equivalence_class]["CFs_ids"]) == 0:
                add_cfs(querylang, cf_alg, g, equivalence_class, (i, j))
            else:
                best_cf, best_diff = get_best_cfs(querylang, g, equivalence_class, (i, j))
                # if min dist>t then geneater new CFs
                if best_diff > t:
                    add_cfs(querylang, cf_alg, g, equivalence_class, (i, j))
                # If max_CF_to_check CFs have been checked, break and use the best so far
                if len(g.nodes[equivalence_class]["CFs_ids_check"][(i,j)]) >= (g.graph['max_CF_to_check']-1):
                    curr_cf = g.graph['CFs_Success'].get((i, j))
                    if curr_cf is not None:
                        curr_diff = g.graph['metric'](get_instance(querylang, g, (i,j)), g.graph['CFs'][curr_cf])
                        best_cf = curr_cf if curr_diff < best_diff else best_cf
                    g.graph['Ans'][(i, j)] = best_cf

        for (i, j) in g.nodes[equivalence_class]['members']:
            if len(g.nodes[equivalence_class]["CFs_ids"]) > 0:
                if g.graph['Ans'].get((i, j)) is None:
                    best_cf, _ = get_best_cfs(querylang, g, equivalence_class, (i, j))
                    g.graph['Ans'][(i, j)] = best_cf

        # associate CFs to each prediction
        for nei in g.neighbors(equivalence_class):
            g.nodes[nei]["CFs_ids"].update(g.nodes[equivalence_class]["CFs_ids"])
    g.graph['total_CFs_generation_time'] = time.time() - start_time
    return g


