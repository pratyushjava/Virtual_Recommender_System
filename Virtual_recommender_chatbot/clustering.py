import sys
import networkx as nx
import random
import gzip
import json
import community
import operator
import numpy






data_path = "../../data/metadata.json.gz"
recommendation_size = 5 
#heuristics constant
#also_bought weight
weight_ab = 0.3
#also viewed weight
weight_av = 0.1
#bought togheter
weight_bt = 1.0


"""
    community_detection technique **********************************************************
"""
def replace_single_quotes(data):
    for i in range(1, len(data)-1):
        if(data[i] == "'" and data[i-1].isalpha() and data[i+1].isalpha()):
            continue
        elif(data[i] == "'"):
            data = data[:i] + '"' + data[i+1:]
            #data[i]'"'
    return data
            
            

def load_graph():
    print("*" * 80)
    print("loading graph!! ")
    print("*" * 80)
    meta_data = gzip.open(data_path, 'r')
    G = nx.Graph()
    for data in meta_data:
        data = data.strip()
        data = data.decode()
        data = replace_single_quotes(data)
        parsed_data = json.loads(data)
        try:
            if 'asin' in parsed_data:
                node = parsed_data['asin']
                if node not in G:
                    G.add_node(node)
                if 'related' in parsed_data:
                    if 'also_bought' in parsed_data['related']:
                        for item in parsed_data['related']['also_bought']:
                            if G.has_edge(node, item):
                                G[node][item]['weight'] = G[node][item]['weight'] + weight_ab
                            else:
                                G.add_edge(node, item, weight=weight_ab)
                    if 'also_viewed' in parsed_data['related']:
                        for item in parsed_data['related']['also_viewed']:
                            if G.has_edge(node, item):
                                G[node][item]['weight'] = G[node][item]['weight'] + weight_av
                            else:
                                G.add_edge(node, item, weight=weight_av)
                    if 'bought_together' in parsed_data['related']:
                        for item in parsed_data['related']['bought_together']:
                            if G.has_edge(node, item):
                                G[node][item]['weight'] = G[node][item]['weight'] + weight_bt
                            else:
                                G.add_edge(node, item, weight=weight_bt)
                    
            
            
        except (RuntimeError, TypeError, NameError):
            print ("exception in loading graph")
            
    meta_data.close()
    print("*" * 80)
    print("graph loaded!! ")
    print("*" * 80)
    return G


def find_community(G):
    #first compute the best partition
    community_graph = community.best_partition(G)
    sorted_community_graph = sorted(community_graph.items(), key=operator.itemgetter(1))
    return sorted_community_graph


def get_target_community(asin, community_graph):
    #find cluster id, if I can't find it return None
    for i in range(len(community_graph)):
        if community_graph[i][0] == asin:
            return community_graph[i][1]
    return None

def community_recommendation(community_graph, community_id, asin, n_recommendation=10):
    target_community = []
    i = 0
    while i < len(community_graph) and community_graph[i][1]<= community_id:
        if community_graph[i][1] == community_id and community_graph[i][0] != asin:
            target_community.append(community_graph[i][0])
        i += 1
    return target_community[:n_recommendation]


def community_detection(asin):
    G = load_graph()
    print("*" * 80)
    print("finding communities in graph!! ")
    print("*" * 80)
    community_graph = find_community(G)
    print("*" * 80)
    print("finding communities in graph done!! ")
    print("*" * 80)
    community_id = get_target_community(asin, community_graph)
    if community_id == None:
        sys.exit("item not found")
        
    community_reccomandations = community_recommendation(community_graph, community_id, asin)
    print("*" * 80)
    print("community detection recommendations are: ")
    print("*" * 80)
    for asin in community_reccomandations:
        print(asin)


"""
    hierarchical clustering technique **********************************************************
"""

def load_data():
    meta_data = gzip.open(data_path, 'r')
    index = 0
    initial_clusters = {}
    relations = {}
    for data in meta_data:
        parsed_data = json.loads(data.decode())
        if 'asin' in parsed_data:
            initial_clusters[parsed_data['asin']] = index
            relations[index] = []
            if 'related' in parsed_data:
                if 'also_bought' in parsed_data['related']:
                    for item in parsed_data['related']['also_bought']:
                        relations[index].append((item, weight_ab))
                if 'also_viewed' in parsed_data['related']:
                    for item in parsed_data['related']['also_viewed']:
                            relations[index].append((item, weight_av))
                if 'bought_together' in parsed_data['related']:
                    for item in parsed_data['related']['bought_together']:
                            relations[index].append((item, weight_bt))
        index += 1
        
    for cluster_index, related_items in relations.items():
        cluster_relations = []
        for i in range(len(related_items)):
            if related_items[i][0] in initial_clusters:
                cluster_relations.append(related_items[i])
        relations[cluster_index] = cluster_relations
        
    meta_data.close()
    return initial_clusters, relations, len(relations)



def initialise_distance_matrix(initial_clusters, relations, n_items):
    distance_matrix = numpy.zeros((n_items, n_items))
    for index,related_items in relations.items():
        for i in range(len(related_items)):
            distance_matrix[index][initial_clusters[related_items[i][0]]] += related_items[i][1]
            
    distance_matrix = distance_matrix + distance_matrix.transpose()
    distance_matrix = numpy.triu(distance_matrix, +1)
    return distance_matrix


def get_next_clusters(distance_matrix, n_items): 
    index = numpy.argmax(distance_matrix)
    row = round(index / n_items)
    column = index % n_items
    return row, column


def unioun_clusters(clusters, matrix, row, col):
    matrix[int(row)][int(col)] = 0
    for i in range(len(matrix[0])):
        if  row < i :
            matrix[int(row)][i] = (matrix[int(row)][i] + max(matrix[int(col)][i], matrix[i][int(col)]))/2
        elif row > i:
            matrix[i][int(row)] = (matrix[i][int(row)] + max(matrix[i][int(col)], matrix[int(col)][i]))/2
        matrix[i][int(col)] = 0
        matrix[int(col)][i] = 0
        
    for pid, cluster_id in clusters.items():
        if cluster_id == col:
            clusters[pid] = row

    return clusters, matrix

def get_hierarchical_clusters(initial_clusters, matrix, n_items):
    index = 0
    max = 1
    while(max > 0.00000005):
        row, col = get_next_clusters( matrix, n_items)
        max = matrix[int(row)][int(col)]
        initial_clusters, matrix = unioun_clusters(initial_clusters, matrix, row, col)
        index += 1

    final_clusters = {}
    for pid, cluster_id in initial_clusters.items():
        if cluster_id not in final_clusters:
            final_clusters[cluster_id] = [pid]
        else:
            final_clusters[cluster_id].append(pid)

    return final_clusters



def get_hierarchical_recommendation(h_clusters, asin, recomendation_size =5):
    cluster_id = -1
    for key, value in h_clusters.items():
        if asin in value:
            cluster_id = key
            break
        
    if cluster_id < 0:
        return -1
    else:
        #filter the list so i don0t return myself
        temp = filter(lambda x: x != asin, h_clusters[cluster_id])
        templst = []
        count = 0
        for i in temp:
            if(count < recomendation_size):
                templst.append(i)
                count += 1
            else:
                break
        return templst



def hierarchical_clustering(asin):

    initial_clusters, relations, n_items = load_data()
    print("*" * 80)
    print("initialise distance matrix ")
    print("*" * 80)
    distance_matrix = initialise_distance_matrix(initial_clusters, relations, n_items)
    print("*" * 80)
    print("computing clusters ")
    print("*" * 80)
    hierarical_clusters = get_hierarchical_clusters(initial_clusters, distance_matrix, n_items)
    print("*" * 80)
    print("computing hierarchical recommendations ")
    print("*" * 80)
    hierarical_based_recommendation = get_hierarchical_recommendation(hierarical_clusters, asin)
    print("*" * 80)
    print("hierarchical clustering recommendations are: ")
    print("*" * 80)
    for asin in hierarical_based_recommendation:
        print(asin)



def main():
    asin = input("Please enter an item ASIN number")
    print("*" * 80)
    print("launching community detection strategy ")
    print("*" * 80)
    community_detection(asin)
    print("*" * 80)
    print("launching hierarchical clustering strategy ")
    print("*" * 80)
    hierarchical_clustering(asin)


        
main()