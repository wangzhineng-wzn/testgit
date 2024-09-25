import networkx as nx
from tqdm import tqdm


def get_neigbors(G, node, depth):
    output = {}
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def Nei_Context(G, L):
    nodelist = list(G.nodes())
    context_dict = {}
    for node in tqdm(nodelist):
        if G.degree(node) == 0:
            context_dict[node] = [node]
        else:
            depth = 1
            flag = True
            while (flag):
                neigh = get_neigbors(G, node, depth)
                if [] in list(neigh.values()):
                    depth = depth - 1
                    break
                summ = 0
                for j in list(neigh.keys()):
                    summ += len(neigh[j])
                if summ >= L - 1:
                    flag = False
                else:
                    depth += 1
            for m in range(1,depth+1):
                nei = neigh[m]
                nei_d = [G.degree(node) for node in nei]
                sort = sorted(range(len(nei_d)), key=lambda k: nei_d[k], reverse=True)
                nei_new = [nei[i] for i in sort]
                neigh[m] = nei_new
            nei_select = []
            for m in range(1,depth+1):
                for n in range(len(neigh[m])):
                    nei_select.append(neigh[m][n])
            nei_select = nei_select[0:L-1]
            nei_select.insert(0, node)
            context_dict[node] = nei_select
    return context_dict

def jaccard_similarity(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    intersection = neighbors1.intersection(neighbors2)
    union = neighbors1.union(neighbors2)
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)

def Sim_Context(G, nei_context_dict):
    nodelist = list(G.nodes())
    context_dict = {}
    for node in tqdm(nodelist):
        nei_context = nei_context_dict[node]
        nei_context_jaccard_dict = {}
        for nodej in nei_context:
            sim = jaccard_similarity(G, node, nodej)
            nei_context_jaccard_dict[nodej] = sim
        sorted_d = sorted(nei_context_jaccard_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = dict(sorted_d)
        context_dict[node] = list(sorted_dict.keys())

    return context_dict