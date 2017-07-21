import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def draw_transmat_graph(G, edge_threshold=0, lw=1, ec='0.2', node_size=15):

    num_states = G.number_of_nodes()

    edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
    edgewidth = np.array(edgewidth)
    edgewidth[edgewidth<edge_threshold] = 0

    labels = {}
    labels[0] = '1'
    labels[1]= '2'
    labels[2]= '3'
    labels[num_states-1] = str(num_states)

    npos=circular_layout(G, scale=1, direction='CW')
    lpos=circular_layout(G, scale=1.15, direction='CW')

    nx.draw_networkx_edges(G, npos, alpha=0.8, width=edgewidth*lw, edge_color=ec)

    nx.draw_networkx_nodes(G, npos, node_size=node_size, node_color='k',alpha=0.8)
    ax = plt.gca()
    nx.draw_networkx_labels(G, lpos, labels, fontsize=18, ax=ax); # fontsize does not seem to work :/

    ax.set_aspect('equal')

    return ax

def graph_from_transmat(transmat):
    G = nx.Graph()
    num_states = transmat.shape[1]
    # make symmetric
    tmat = (transmat + transmat.T) / 2

    for s1 in range(num_states):
        for s2 in range(num_states):
            G.add_edge(s1, s2, weight=tmat[s1,s2])

    return G

def _process_params(G, center, dim):
    # Some boilerplate code.

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def rescale_layout(pos, scale=1):
    """Return scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(pos[:, i].max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos

def circular_layout(G, scale=1, center=None, dim=2, direction='CCW'):
    # dim=2 only
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph or list of nodes

    scale : float
        Scale factor for positions

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout, currently only dim=2 is supported

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.circular_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    G, center = _process_params(G, center, dim)

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        if direction == 'CCW':
            pos = np.column_stack([np.cos(theta), np.sin(theta)])
        else:
            pos = np.column_stack([np.sin(theta), np.cos(theta)])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos