# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:29 2016

@author: Laura Drummer
"""
import networkx as nx
import pandas as pd
import config
import operator
# import scipy.sparse.linalg.eigen.arpack as eigenerror


def dict_flattener(uid, email_dict):
    """Turns nested art-network dictionary into a flat list of dictionaries.

    Keyword arguments:
    uid -- message_id correlating to a unique email message
    email_dict -- nested dictionary with informaiton on the various author -
                  recipient pairs and their communication type (to,cc,bcc)
    """
    flat_list = []
    for key in email_dict['ar_pairs'].keys():
        for ar_pair in email_dict['ar_pairs'][key]['pairs']:
            flat_dict = {}
            flat_dict['message_id'] = uid
            try:
                flat_dict['from'], flat_dict['to'] = ar_pair.split('___')
                flat_dict['weight'] = email_dict['ar_pairs'][key]['weight']
            except ValueError:
                return(key)
            flat_dict['to_type'] = key
            flat_dict['ar_pair'] = ar_pair
            flat_list.append(flat_dict)
    return(flat_list)


def caluclate_network_attributes(Graph):
    """Caluclates several graph attributes and stores them in a dataframe.

    Keyword arguments:
    Graph -- DiGraph
    """
    # Reads in a list of users from the config file - limits caluclations to
    # most interesting user.  This is very resource intensive.
    all_nodes = config.interesting_users
    graph_df = pd.Series(all_nodes).to_frame(name="Node")
    degree = nx.degree(Graph)
    degree_cen = nx.degree_centrality(Graph)
    in_degree = nx.in_degree_centrality(Graph)
    out_degree = nx.out_degree_centrality(Graph)
    closeness = nx.closeness_centrality(Graph)
    between = nx.betweenness_centrality(Graph)
    try:
        eigen = nx.eigenvector_centrality_numpy(Graph)
        graph_df['Eigenvector'] = pd.Series([eigen[node]
                                             for node in all_nodes])
    # except eigenerror.ArpackNoConvergence:
    except:
        # If no eigenvector can be caluclated, set all to zero.
        graph_df['Eigenvector'] = pd.Series([0 for n in all_nodes])
    graph_df['Degree'] = pd.Series(
        [degree[node] for node in all_nodes if node in degree.keys()])
    graph_df['DegreeCentrality'] = pd.Series([degree_cen[node]
                                              for node in all_nodes
                                              if node in degree_cen.keys()])
    graph_df['InDegree'] = pd.Series([in_degree[node]
                                      for node in all_nodes
                                      if node in in_degree.keys()])
    graph_df['OutDegree'] = pd.Series([out_degree[node]
                                       for node in all_nodes
                                       if node in out_degree.keys()])
    graph_df['Closeness'] = pd.Series([closeness[node]
                                       for node in all_nodes
                                       if node in closeness.keys()])
    graph_df['Betweeness'] = pd.Series([between[node]
                                        for node in all_nodes
                                        if node in between.keys()])

    return(graph_df)


def get_max_AR_pairs(digraph):
    """Parses a diGraph and returns the top three most communicative pairs.

    Keyword arguments:
    digraph -- any directional graph created in networkx
    """
    ar_weight_dict = {}
    for edge in digraph.edges():
        a, r = edge
        ar_weight_dict[(a, r)] = digraph[a][r]['weight']
    sorted_dict = sorted(ar_weight_dict.items(), key=operator.itemgetter(1))
    top = sorted_dict[-3:]
    top.reverse()
    return(top)
