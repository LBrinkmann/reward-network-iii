import hashlib
import json
import random
import string
import yaml
from collections import Counter
import argparse

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from common.models.network import Network, Node, Edge
from common.models.environment import Environment


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


class NetworkGenerator:
    """
    Network Generator class
    """

    def __init__(self, environment: Environment):
        """
        Initializes a network generator object with parameters
        obtained from streamlit form
        Args:
            params (dict): parameter dictionary
        """

        self.network_objects = []
        self.networks = []
        self.env = environment

        # parameters for visualization
        # This parameter is used to determine the size of the nodes to get proper coordinates for edges
        self.node_size = 2200
        self.arc_rad = 0.1

        self.from_to = {
            (e_def.from_level, tl): e_def.rewards
            for e_def in self.env.edges
            for tl in e_def.to_levels
        }
        self.start_node = None

    def generate(self, n_networks):
        """
        Using the functions defined above this method generates networks with
        visualization info included.
        The generated network(s) are also saved in a json file at location
        specified by save_path
        """

        # sample and store training networks
        self.networks = []
        for _ in range(n_networks):
            g = self.sample_network()
            net = nx.json_graph.node_link_data(g)

            # NEW: shuffle randomly the order of the nodes in circular layout
            pos = nx.circular_layout(g)

            pos_map = {
                n: {"x": p[0] * 100, "y": p[1] * -1 * 100} for n, p in pos.items()
            }

            # NEW: add vertices for visualization purposes
            plt.figure()
            plt.axis("equal")
            arrow_size = 0.1
            node_size = self.node_size

            _ = nx.draw_networkx_nodes(g, pos=pos, node_size=node_size)

            for ii, e in enumerate(g.edges()):
                if reversed(e) in g.edges():
                    net["links"][ii]["arc_type"] = "curved"
                    arc = nx.draw_networkx_edges(
                        g,
                        pos,
                        edgelist=[e],
                        node_size=node_size,
                        arrowsize=arrow_size,
                        connectionstyle=f"arc3, rad = {self.arc_rad}",
                    )
                else:
                    net["links"][ii]["arc_type"] = "straight"
                    arc = nx.draw_networkx_edges(
                        g,
                        pos,
                        edgelist=[e],
                        node_size=node_size,
                        arrowsize=arrow_size,
                    )

                vert = arc[0].get_path().vertices.T[:, :3] * 100

                net["links"][ii]["source_x"] = vert[0, 0]
                net["links"][ii]["source_y"] = -1 * vert[1, 0]
                net["links"][ii]["arc_x"] = vert[0, 1]
                net["links"][ii]["arc_y"] = -1 * vert[1, 1]
                net["links"][ii]["target_x"] = vert[0, 2]
                net["links"][ii]["target_y"] = -1 * vert[1, 2]

            plt.close("all")

            network_id = hashlib.md5(
                json.dumps(net, sort_keys=True).encode("utf-8")
            ).hexdigest()

            c = Counter([e["source"] for e in net["links"]])

            if (
                (
                    sum(value for value in c.values())
                    == int(self.env.n_edges_per_node * self.env.n_nodes)
                )
                and (
                    min(value for value in c.values()) == int(self.env.n_edges_per_node)
                )
                and len(list(c.keys())) == self.env.n_nodes
            ):
                create_network = self.create_network_object(
                    pos_map=pos_map,
                    n_steps=self.env.n_steps,
                    network_id=network_id,
                    **net,
                )
                self.networks.append(create_network)
                print(f"Network {len(self.networks)} created")
            else:
                print(
                    f"counter {c}, nodes are {list(c.keys())} "
                    f"(n={len(list(c.keys()))})"
                )

        return self.networks

    def get_source_reward_idx(self, G, source_node):
        return [
            G.edges[source_node, target_node]["reward_idx"]
            for target_node in G[source_node]
        ]

    # individual network building functions
    #######################################
    def add_link(self, G, source_node, target_node):
        from_level = G.nodes[source_node]["level"]
        to_level = G.nodes[target_node]["level"]
        possible_rewards = self.from_to[(from_level, to_level)]
        other_source_reward_idx = self.get_source_reward_idx(G, source_node)
        if len(possible_rewards) > len(other_source_reward_idx):
            possible_rewards = [
                r for r in possible_rewards if r not in other_source_reward_idx
            ]
        reward_idx = random.choice(possible_rewards)
        reward = self.env.rewards[reward_idx]
        G.add_edge(
            source_node,
            target_node,
            reward=reward.reward,
            reward_idx=reward_idx,
            color=reward.color,
        )

    @staticmethod
    def add_new_node(G, level):
        idx = len(G)
        name = string.ascii_uppercase[idx % len(string.ascii_lowercase)] + str(level)
        G.add_node(idx, name=name, level=level)
        return idx

    @staticmethod
    def nodes_random_sorted_by_in_degree(G, nodes):
        return sorted(
            nodes, key=lambda n: G.in_degree(n) + random.random() * 0.1, reverse=False
        )

    @staticmethod
    def nodes_random_sorted_by_out_degree(G, nodes):
        return sorted(
            nodes,
            key=lambda n: G.nodes[n]["level"]
            + G.out_degree(n) * 0.1
            + random.random() * 0.01,
            reverse=False,
        )

    def edge_is_allowed(self, G, source_node, target_node):
        if source_node == target_node:
            return False
        if target_node in G[source_node]:
            return False
        from_level = G.nodes[source_node]["level"]
        to_level = G.nodes[target_node]["level"]
        if (from_level, to_level) not in self.from_to:
            return False
        # other source target levels
        other_source_target_levels = [G.nodes[n]["level"] for n in G[source_node]]
        if len(other_source_target_levels) == 1 and max(other_source_target_levels) > 0 and (to_level in other_source_target_levels):
            return False
        return True

    def allowed_target_nodes(self, G, nodes, source_node):
        return [node for node in nodes if self.edge_is_allowed(G, source_node, node)]

    def allowed_source_nodes(self, G):
        min_degree = min(G.out_degree(n) for n in G.nodes)
        return [
            n
            for n in G.nodes
            if (G.out_degree(n) < self.env.n_edges_per_node)
            and (G.out_degree(n) <= min_degree)
        ]

    def assign_levels(self, graph):
        levels = self.env.levels.copy()
        # min number of nodes to each level
        for level in levels:
            level.n_nodes = level.min_n_nodes

        # total number of nodes per levels
        n_nodes = sum([level.n_nodes for level in levels])
        assert n_nodes <= self.env.n_nodes

        # spread missing nodes over levels
        for i in range(self.env.n_nodes - n_nodes):
            # possible levels
            possible_levels = [
                level
                for level in levels
                if (level.max_n_nodes is None) or (level.n_nodes < level.max_n_nodes)
            ]
            # choose level
            level = random.choice(possible_levels)
            # add node to level
            level.n_nodes += 1

        # add nodes to graph
        level_list = [level.idx for level in levels for _ in range(level.n_nodes)]
        random.shuffle(level_list)
        for level in level_list:
            self.add_new_node(graph, level)
        zero_level_nodes = [node for node in graph if graph.nodes[node]["level"] == 0]
        self.start_node = random.choice(zero_level_nodes)

    def sample_network(self):
        graph = nx.DiGraph()

        self.assign_levels(graph)
        for i in range(int(self.env.n_edges_per_node * self.env.n_nodes)):
            allowed_source_nodes = self.allowed_source_nodes(graph)
            if len(allowed_source_nodes) == 0:
                raise ValueError("No allowed nodes to connect from.")
            source_node = self.nodes_random_sorted_by_out_degree(
                graph, allowed_source_nodes
            )[0]
            allowed_target_nodes = self.allowed_target_nodes(
                graph, graph.nodes, source_node
            )
            if len(allowed_target_nodes) == 0:
                raise ValueError(
                    f"No allowed nodes to connect to. Source level: {graph.nodes[source_node]['level']}"
                )
            target_node = self.nodes_random_sorted_by_in_degree(
                graph, allowed_target_nodes
            )[0]
            self.add_link(graph, source_node, target_node)

        return graph

    @staticmethod
    def parse_node(name, pos_map, level, id, starting_node, **__):
        return Node(
            node_num=id,
            display_name=name,
            node_size=3,
            level=level,
            starting_node=starting_node,
            **pos_map[id],
        )

    @staticmethod
    def parse_link(source, target, **props):
        return Edge(source_num=source, target_num=target, **props)

    def create_network_object(self, pos_map, *, nodes, links, network_id, **kwargs):
        return Network(
            nodes=[
                self.parse_node(
                    **n, pos_map=pos_map, starting_node=n == self.start_node
                )
                for n in nodes
            ],
            edges=[self.parse_link(**l) for l in links],
            starting_node=self.start_node,
            network_id=network_id,
        )

    def save_as_json(self):
        return json.dumps([n.dict() for n in self.networks])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate networks based on environment parameters."
    )
    parser.add_argument(
        "-i", "--input", help="Path to the input YAML file", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output JSON file", required=True
    )
    args = parser.parse_args()

    environment = load_yaml(args.input)
    seed = environment["seed"]
    random.seed(seed)
    np.random.seed(seed)

    generate_params = Environment(**environment)

    net_generator = NetworkGenerator(generate_params)
    networks = net_generator.generate(environment["n_networks"])
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(net_generator.save_as_json()))
