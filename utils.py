import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt

def draw(G, m, file_name="network", show_edges=False, show_fig=False):
    G = deepcopy(G)
    for node in G.nodes:
        G.nodes[node]["dynamic"] = G.nodes[node]["dynamic"] / m
    pos = {node:node for node in G.nodes}
    node_colors = {}
    node_shapes = {}
    node_shape_chars = "Xos^>v<dph8"
    for node in G.nodes:
        char = node_shape_chars[G.nodes[node]["static"]]
        if char not in node_shapes.keys():
            node_shapes[char] = []
        node_shapes[char].append(node)
        if char not in node_colors.keys():
            node_colors[char] = []
        node_colors[char].append(G.nodes[node]["dynamic"])
    for node_shape in node_shapes.keys():
        nx.draw_networkx_nodes(G, pos=pos, nodelist=node_shapes[node_shape], node_shape=node_shape, node_color=node_colors[node_shape])
    if show_edges:
        nx.draw_networkx_edges(G, pos=pos)
    if show_fig:
        plt.show()
    else:
        if file_name[-4:] != ".pdf":
            file_name += ".pdf"
        plt.savefig(file_name, bbox_inches='tight')

def grid_2d_moore_graph(m, n, periodic=False):
        
    m = int(m)

    n = int(n)
    
    G = nx.empty_graph(0,None)
    G.name = "grid_2d_moore_graph"
    rows = range(m)
    columns = range(n)
    G.add_nodes_from( (i,j) for i in rows for j in columns )
    G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in columns if i>0 )
    G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in columns if j>0 )

    G.add_edges_from( ((i,j),(i-1,j-1)) for i in rows for j in columns if j>0 and i>0 )
    G.add_edges_from( ((i,j),(i-1,j+1)) for i in rows for j in columns if i>0 and j<n-1 )

    if periodic:
        if n>2:
            G.add_edges_from( ((i,0),(i,n-1)) for i in rows )
            G.add_edges_from( ((i,0),(i-1,n-1)) for i in rows if i>0)
            G.add_edges_from( ((i,0),(i+1,n-1)) for i in rows if i<n-1)

        if m>2:
            G.add_edges_from( ((0,j),(m-1,j)) for j in columns )
            G.add_edges_from( ((0,j),(m-1,j-1)) for j in columns if j>0)
            G.add_edges_from( ((0,j),(m-1,j+1)) for j in columns if j<m-1)

        #Diagonal to diagonal
        G.add_edge( (0,0),(m-1,n-1) )
        G.add_edge( (m-1,0),(0,n-1) )

        G.name = "periodic_grid_2d_graph(%d,%d)"%(m,n)
    return G
