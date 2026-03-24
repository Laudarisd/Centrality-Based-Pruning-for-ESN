# -*- coding: utf-8 -*-
"""
Reference: Brandes (2001), A Faster Algorithm for Betweenness Centrality.
Formula: c_B(v) = sum_{s,t in V} sigma(s,t|v)/sigma(s,t), and similarly for edges.
"""

from heapq import heappop, heappush
from itertools import count
import logging
import random

__all__ = ["betweenness_centrality", "edge_betweenness_centrality", "edge_betweenness"]

logger = logging.getLogger(__name__)


# Compute shortest-path betweenness centrality for nodes.
def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    logger.debug(
        "betweenness_centrality called | nodes=%d directed=%s k=%s normalized=%s weight=%s endpoints=%s",
        len(G),
        G.is_directed(),
        k,
        normalized,
        weight,
        endpoints,
    )
    betweenness = dict.fromkeys(G, 0.0)

    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(list(G.nodes()), k)

    for s in nodes:
        if weight is None:
            S, P, sigma = _single_source_shortest_path_basic(G, s)
        else:
            S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)

        if endpoints:
            betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness = _accumulate_basic(betweenness, S, P, sigma, s)

    return _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        k=k,
        endpoints=endpoints,
    )


# Compute shortest-path betweenness centrality for edges.
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    logger.debug(
        "edge_betweenness_centrality called | nodes=%d directed=%s k=%s normalized=%s weight=%s",
        len(G),
        G.is_directed(),
        k,
        normalized,
        weight,
    )
    betweenness = dict.fromkeys(G, 0.0)
    betweenness.update(dict.fromkeys(G.edges(), 0.0))

    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(list(G.nodes()), k)

    for s in nodes:
        if weight is None:
            S, P, sigma = _single_source_shortest_path_basic(G, s)
        else:
            S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)

        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)

    for n in G:
        del betweenness[n]

    return _rescale_e(betweenness, len(G), normalized=normalized, directed=G.is_directed())


# Backward-compatible alias.
def edge_betweenness(G, k=None, normalized=True, weight=None, seed=None):
    return edge_betweenness_centrality(G, k, normalized, weight, seed)


# Single-source BFS shortest paths helper.
def _single_source_shortest_path_basic(G, s):
    S = []
    P = {v: [] for v in G}
    sigma = dict.fromkeys(G, 0.0)
    D = {}

    sigma[s] = 1.0
    D[s] = 0
    Q = [s]

    while Q:
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]

        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)

    return S, P, sigma


# Single-source Dijkstra shortest paths helper.
def _single_source_dijkstra_path_basic(G, s, weight):
    S = []
    P = {v: [] for v in G}
    sigma = dict.fromkeys(G, 0.0)
    D = {}

    sigma[s] = 1.0
    seen = {s: 0}
    c = count()
    Q = []
    heappush(Q, (0, next(c), s, s))

    while Q:
        dist, _, pred, v = heappop(Q)
        if v in D:
            continue

        sigma[v] += sigma[pred]
        S.append(v)
        D[v] = dist

        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                heappush(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma


# Accumulate node betweenness without endpoint contribution.
def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


# Accumulate node betweenness with endpoint contribution.
def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness


# Accumulate edge betweenness.
def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


# Rescale node betweenness values.
def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    if normalized:
        if endpoints:
            scale = None if n < 2 else 1 / (n * (n - 1))
        elif n <= 2:
            scale = None
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:
        scale = 0.5 if not directed else None

    if scale is not None:
        if k is not None:
            scale *= n / k
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness


# Rescale edge betweenness values.
def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        scale = None if n <= 1 else 1 / (n * (n - 1))
    else:
        scale = 0.5 if not directed else None

    if scale is not None:
        if k is not None:
            scale *= n / k
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness
