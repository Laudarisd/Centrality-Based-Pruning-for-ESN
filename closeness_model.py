# -*- coding: utf-8 -*-
"""
Reference: Freeman (1979), Centrality in networks: I. Conceptual clarification.
Formula: C(u) = (n-1) / sum_{v=1}^{n-1} d(v,u), with optional WF scaling ((n-1)/(N-1)).
"""

import functools
import logging

import networkx as nx
import numpy as np

__all__ = ["closeness_centrality"]

logger = logging.getLogger(__name__)


# Compute node closeness centrality with optional weighted distance and WF scaling.
def closeness_centrality(G, u=None, distance=None, wf_improved=True, reverse=False):
    logger.debug(
        "closeness_centrality called | nodes=%d directed=%s distance=%s wf_improved=%s reverse=%s",
        len(G),
        G.is_directed(),
        distance,
        wf_improved,
        reverse,
    )
    if distance is not None:
        path_length = functools.partial(nx.single_source_dijkstra_path_length, weight=distance)
    else:
        if G.is_directed() and not reverse:
            path_length = nx.single_target_shortest_path_length
        else:
            path_length = nx.single_source_shortest_path_length

    nodes = G.nodes() if u is None else [u]
    result = {}

    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = float(np.sum(np.array(list(sp.values()))))

        if totsp > 0.0 and len(G) > 1:
            result[n] = (len(sp) - 1.0) / totsp
            if wf_improved:
                result[n] *= (len(sp) - 1.0) / (len(G) - 1)
        else:
            result[n] = 0.0

    if u is not None:
        return result[u]
    return result
