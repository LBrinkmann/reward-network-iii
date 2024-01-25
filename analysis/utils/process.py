from typing import List, Optional
import numpy as np



def eval_move(edge, all_edges):
    max_reward = max([e['reward'] for e in all_edges])
    min_reward = min([e['reward'] for e in all_edges])

    myopic = edge['reward'] == max_reward
    has_large_loss = min_reward == -50
    large_loss_taken = edge['reward'] == -50

    optimal = large_loss_taken if has_large_loss else myopic

    return {
        'myopic': myopic,
        'optimal': optimal,
        'large_loss_taken': large_loss_taken,
    }



def process_moves(network: dict, moves_nodes: List[int], correct_repeats: Optional[List[bool]] = None):
    edges_by_source = {}
    for edge in network['edges']:
        edges_by_source.setdefault(edge['source_num'], []).append(edge)

    edges_by_source_target = {
        source_num: {
            edge['target_num']: edge
            for edge in edges
        }
        for source_num, edges in edges_by_source.items()
    }


    nodes_by_num = {
        node['node_num']: node
        for node in network['nodes']
    }

    moves = []
    for i, (source_num, target_num) in enumerate(zip(moves_nodes[:-1], moves_nodes[1:])):
        edges = edges_by_source_target[source_num]
        edge = edges[target_num]
        moves.append({
            "move_idx": i,
            "source_num": source_num,
            "target_num": target_num,
            "reward": edge['reward'],
            "level": nodes_by_num[target_num]['level'],
            "correct_repeat": correct_repeats[i] if correct_repeats is not None else None,
            **eval_move(edge, edges.values())
        })
    return moves

def process_solution(network: dict, solution: dict):
    moves = process_moves(network, solution['moves'], solution['correctRepeats'])
    aggregate = {
        'myopic': np.sum([m['myopic'] for m in moves]),
        'optimal': np.sum([m['optimal'] for m in moves]),
        'large_loss_taken': np.sum([m['large_loss_taken'] for m in moves]),
        'score': solution['score'],
        'n_moves': len(moves),
    }
    return aggregate
