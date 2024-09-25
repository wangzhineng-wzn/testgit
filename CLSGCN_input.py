from Utils import *
import numpy as np


# CLSGCN_input_1
def fea1(G, L):
    fea_dict = {}
    nei_context_dict = Nei_Context(G, L)
    for node in tqdm(list(nei_context_dict.keys())):
        context = nei_context_dict[node]
        L_real = len(context)

        if L_real == 1:
            fea_dict[node] = np.zeros((1, L, L))
        else:
            # adj
            adj = np.zeros((L_real,L_real))
            for i in range(L_real):
                for j in range(L_real):
                    if G.has_edge(context[i],context[j]):
                        adj[i][j] = 1
            # weight_adj
            edj = np.zeros((L_real,L_real))
            for i in range(L_real):
                for j in range(L_real):
                    if G.has_edge(context[i],context[j]):
                        edj[i][j] = 1 + np.log2(G.degree(context[i]))
            # static_matrix
            fdj1 = edj.copy()
            for i in range(L_real):
                fdj1[i][i] = G.degree(context[i])
            for i in range(1,L_real):
                fdj1[0][i] = edj[0][i] * G.degree(context[i])
                fdj1[i][0] = edj[i][0] * G.degree(context[i])
            # condensed_static_matrix
            sdj = fdj1.copy()
            nei1 = get_neigbors(G, node, 2)[1]
            nei2 = get_neigbors(G, node ,2)[2]
            for i in nei1:
                if i in context:
                    i_in_lnei_index = context.index(i)
                    sdj[:, i_in_lnei_index] = 0
                    sdj[i_in_lnei_index, :] = 0
            for j in nei2:
                if j in context:
                    j_in_lnei_index = context.index(j)
                    sdj[0][j_in_lnei_index] = 1
                    sdj[j_in_lnei_index][0] = 1

            # dynamic_matrix
            fdj2 = np.zeros(shape=(L_real,L_real))
            for i in range(L_real):
                for j in range(L_real):
                    fdj2[i][j] = abs(fdj1[i][j] - sdj[i][j])

            # sturcture_feature_matrix
            fdj = np.zeros(shape=(1,L,L))
            fdj[0,0:L_real,0:L_real] = fdj2

            fea_dict[node] = fdj

    return fea_dict, nei_context_dict

# CLSGCN_input_2
def fea2(G, L, nei_context_dict):
    DC = dict(nx.degree(G))
    BC = dict(nx.betweenness_centrality(G))
    CC = dict(nx.closeness_centrality(G))
    CL = dict(nx.clustering(G))
    fea_dict = {}
    sim_context_dict = Sim_Context(G, nei_context_dict)
    for node in tqdm(list(sim_context_dict.keys())):
        context = sim_context_dict[node]
        L_real = len(context)
        F = np.zeros((L_real,4))
        for i in range(L_real):
            F[i][0] = (DC[context[i]]-min(DC.values()))/(max(DC.values())-min(DC.values()))
            F[i][1] = (BC[context[i]]-min(BC.values()))/(max(BC.values())-min(BC.values()))
            F[i][2] = (CC[context[i]]-min(CC.values()))/(max(CC.values())-min(CC.values()))
            F[i][3] = (CL[context[i]]-min(CL.values()))/(max(CL.values())-min(CL.values()))
        F_result = np.zeros((L,4))
        F_result[:L_real] = F
        fea_dict[node] = F_result

    return fea_dict
