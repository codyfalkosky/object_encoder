import numpy as np


# +
def _cosine_similiarty_np(x):
    '''
    given a matrix a, computes cosine similarity of all vectors in a to all other vectors in a

    Args:
        x (tensor) : matrix of row vectors for comparison
    Returns:
        sim_matrix (array) : matrix showing the cosine similiarity of every vector to every other
    '''
    x          = np.array(x)
    
    # normalize all vectors to magnitude 1
    norm       = np.linalg.norm(x, axis=1, keepdims=True)
    x_n        = x / norm

    # calculate cosine similarity
    sim_matrix = x_n @ x_n.T

    return sim_matrix

def _euclidean_distance(self, x):
    
    '||p - q||^2 = ||p||^2 + ||q||^2 - 2 * p dot q.T '
    
    # compute parts
    squared_norms = tf.reduce_sum(tf.square(x), axis=1)

    p_sn = tf.reshape(squared_norms, [-1, 1])
    q_sn = tf.reshape(squared_norms, [1 ,-1])

    # ||p||^2 + ||q||^2 - 2 * p * q.T
    squared_distance = (p_sn + q_sn) - 2 * tf.matmul(x, x, transpose_b=True)

    # clip for floating point stability
    squared_distance = tf.maximum(squared_distance, 0.)

    # square root for distance
    distance = tf.sqrt(squared_distance)

    return dist_matrix

def _as_nodes(edges):
    '''
    convert list of pairs to a "graph" of "nodes" and "connections" represented with a dictionary

    Args:
        edges (array) : 2d like [[x,y], [x,y], [x,y]]
            a list of pairs with each pair representing a single connection, pairs repeat for many connections
    Returns:
        graph (dictionary) : where every key is a node and values are a dictionary where keys are the nodes they
            connect to and values are "number of connections" or "strength of connection"
    '''
    graph = {}

    for edge in edges:
        x, y = edge

        if x not in graph:
            graph[x] = {}

        if y not in graph:
            graph[y] = {}

        if y not in graph[x]:
            graph[x][y] = 0       

        graph[x][y] += 1

        if x not in graph[y]:
            graph[y][x] = 0

        graph[y][x] += 1

    return graph

def _sort_nodes(graph):
    '''
    given a graph return all nodes in numerical order

    Args:
        graph (dictionary) : output of _as_nodes
    Returns:
        nodes (list) : [0, 1, 2, 3, ...] all nodes in order
    '''
    nodes         = []
    n_connections = []
    
    for k, v in graph.items():
        nodes.append(k)
        n_connections.append(sum(v.values()))
        

    order = np.argsort(n_connections)[::-1]
    nodes = np.array(nodes)
    nodes = nodes[order]
    return nodes.tolist()


def _find_edge_strength_percentile(graph, node, percentile=.3):
    '''
    will start a 'node' in 'graph' and go through all connections and record strength, then returns
    strength at percentile

    Args:
        graph (dictionary) : output of _as_nodes
        node (int) : node to explore
        percentile (float) : percentile of connection strengths found in cluster connected to node
    Returns:
        strength_at_percentile (int) : connection strength at percentile (for _gather_node_set threshold)
    '''
    checked   = set()
    strengths = []
    to_check  = {node}

    while to_check:

        current_node = to_check.pop()

        if current_node not in checked:
            checked.add(current_node)

            to_check.update(set(graph[current_node].keys()))

            strengths.extend(list(graph[current_node].values()))

    strengths.sort()

    idx = round(  (len(strengths) - 1) * percentile  )

    if idx < 0:
        print('NO STRENGTHS FOUND')

    return strengths[idx]

def _gather_node_set(graph, node, threshold):
    '''
    gathers all nodes connected to a node whos connection strength is >= threshold

    Args:
        graph (dictionary) : output of _as_nodes
        node (int) : node to explore
        threshold (int) : minimum valid connection strength
    Returns:
        node_set (set) : all nodes connected to node
    
    '''
    node_set = set()
    to_check = {node}

    while to_check:

        current_node = to_check.pop()

        if current_node not in node_set:
            node_set.add(current_node)

            for item in graph[current_node].items():
                possible_connection = item[0]
                connection_strength = item[1]

                if connection_strength >= threshold:
                    if possible_connection not in node_set:
                        to_check.add(possible_connection)

    return node_set


def _extract_node_sets(graph, nodes, percentile):
    '''
    takes a list of nodes and parses them into node_sets based on a graph

    Args:
        graph (dictionary) : output of _as_nodes
        nodes (list) : list of all nodes (should be sorted by _sort_nodes)
    Returns:
        node_sets (list of sets) : list of all node sets, where sets contain clustered nodes
    '''
    node_sets      = []
    assigned_nodes = set()
    
    for node in nodes:
        if node not in assigned_nodes:
            threshold = _find_edge_strength_percentile(graph, node, percentile)
            node_set  = _gather_node_set(graph, node, threshold)

            node_sets.append(node_set)
            assigned_nodes.update(node_set)

    return node_sets

def _node_sets_to_labels(node_sets, nodes):
    '''
    final stage, converting node_sets and nodes to labels

    Args:
        node_sets (list of sets) : list of all node sets
    
    '''
    nodes = sorted(nodes)

    label_map = {}

    for label, node_set in enumerate(node_sets):
        for node in node_set:
            label_map[node] = label

    labels = [label_map[node] for node in nodes]

    return labels
    


# -

def decode_to_labels(model_output, similarity_threshold=.99, percentile=.3, decode_basis='cosine', euclidean_thresh=.5):
    '''
    decodes object_encoder model_output to labels

    Args:
        model_output (tensor) : shape (n_encodings, encoding_dim)
        similarity_threshold (float) : minimum cosine similary threshold for considering two encodings similar
        percentile (float) : percentile of cluster connection strength to consider a connection valid
    Return:
        labels (list) : every vector labeled by cluster
    '''
    if decode_basis == 'cosine':
        sim_matrix = _cosine_similiarty_np(model_output)
        graph      = _as_nodes(np.argwhere(sim_matrix > similarity_threshold))
    elif decode_basis == 'euclidean':
        dist_matrix= _euclidean_distance(model_output)
        graph      = _as_nodes(np.argwhere(dist_matrix < euclidean_thresh))
    nodes      = _sort_nodes(graph)
    node_sets  = _extract_node_sets(graph, nodes, percentile)
    labels     = _node_sets_to_labels(node_sets, nodes)

    return labels






