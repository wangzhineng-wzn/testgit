from Utils import *
from networkx.algorithms.community import asyn_lpa_communities as lpa
import numpy as np


# RDPP
def RDPP(G, score_list, p):
    community_list = list(lpa(G))
    nodelist = list(G.nodes())
    merged = zip(nodelist, score_list)
    score_dict = {k: v for k, v in merged}
    num_seed = round(len(G.nodes()) * p)
    seed_list = []
    while len(seed_list) < num_seed:
        seed = max(score_dict.items(), key=lambda x: x[1])[0]
        for community in community_list:
            if seed in community:
                seed_community = community
        seed_list.append(seed)
        del(score_dict[seed])
        nei_dict = get_neigbors(G, seed, depth=2)
        nei_1 = nei_dict[1]
        nei_2 = nei_dict[2]
        for nei in nei_1+nei_2:
            if nei not in seed_list:
                paths = nx.all_simple_paths(G, seed, nei, cutoff=2)
                notk_list = []
                for path in paths:
                    if len(path) == 3:
                        k1 = 0.2 + 1 / G.degree(path[0])
                        k2 = 0.2 + 1 / G.degree(path[1])
                        notk = 1 - k1*k2
                        notk_list.append(notk)
                    if len(path) == 2:
                        k1 = 0.2 + 1 / G.degree(path[0])
                        notk = 1 - k1
                        notk_list.append(notk)

                if nei not in seed_community:
                    INF = 1 - np.prod(notk_list)
                else:
                    sim = jaccard_similarity(G, nei, seed)
                    INF = 1 - (1-sim) * np.prod(notk_list)
                score_dict[nei] = score_dict[nei] * (1 - INF)

    return seed_list