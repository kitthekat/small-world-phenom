"""
Christopher Kramer

This Module uses Python's builtin Queue package for the MyQueue class.

Runs much faster.
"""

from typing import List
from collections import defaultdict, deque, Counter
from itertools import repeat
import queue
import pandas as pd
from functools import reduce
from datetime import datetime
import warnings

FilePath = str
GraphData = defaultdict
Vertex = int
Distances = List


class MyQueue(queue.Queue):
    """
    Queue class inheriting from queue std library package
    """
    def __init__(self):
        super().__init__()

    def __len__(self):
        """
        Magic method to return length of queue
        :return:
        """
        return self.qsize()

    def __str__(self):
        """
        Returns string representation of queue without dequeueing elements
        :return:
        """
        with self.mutex:
            return str(list(self.queue))

    def dequeue(self):
        """
        Returns next queue element if queue is not empty
        :return:
        """
        return self.get() if not self.empty() else None


def enqueue(q: MyQueue, s: Vertex): q.put(s)  # enqueue utility function to match pseudocode


def loadGraph(edge_file_path: FilePath = 'edges.txt') -> GraphData:
    """
    Returns a default dictionary of vertex : [neighbors] pairs
    :param edge_file_path:
    :return:
    """
    def to_int(l: list): return [int(x) for x in l]  # Converts the elements of a list to integer

    def adj_updater(d, edge):  # Updates vertex : neighbor pairs and the complement (neighbor : vertex)
        d[edge[0]].add(edge[1])
        d[edge[1]].add(edge[0])
    with open(edge_file_path, 'r') as fp:
        # Creates a list of [A, B] integer pairs representing node connections
        edges = list(map(to_int, filter(lambda x: len(x) == 2, [x.split(' ') for x in fp.read().split('\n')])))
    adj_list = defaultdict(lambda: set())
    # Converts node connection list to adjacency dictionary (node: connections)
    deque(map(adj_updater, repeat(adj_list), edges))
    return adj_list


def _BFS_preliminary_attempt(G: GraphData, s: Vertex) -> Distances:
    """
    Initial attempt at BFS algo
    :param G:
    :param s:
    :return:
    """
    # Initialize an empty queue Q
    q = MyQueue()
    # for each u \in V do d_u <- \infty
    distances = [None for _ in range(len(G))]
    # ds <- 0
    ds = 0
    # Enqueue(Q, s)
    enqueue(q, s)
    # while Q is not empty
    while q.queue:
        # do u <- dequeue(Q)
        u = q.dequeue()
        # for each neighbor v of u
        for v in G[u]:
            # do if d_v = \infty
            if distances[v] is None:
                # then d_v <- d_u + 1
                distances[v] = ds + 1
                # enqueue(Q, v)
                enqueue(q, v)
        ds += 1
    distances[s] = 0  # sometimes the starting vertex gets a non-0 number
    # return d
    return distances


def BFS(G: GraphData, s: Vertex) -> Distances:
    """
    Second attempt at BFS algo
    :param G:
    :param s:
    :return:
    """
    # Initialize an empty queue Q
    q = MyQueue()
    # for each u \in V do d_u <- \infty
    d = defaultdict(lambda: None)
    # ds <- 0
    d[s] = 0
    # Enqueue(Q, s)
    enqueue(q, s)
    # while Q is not empty
    while q.queue:
        # do u <- dequeue(Q)
        u = q.dequeue()
        # d_u + 1
        new_d = d[u] + 1
        # for each neighbor v of u
        # do if d_v = \infty
        for v in G[u].difference(d):
            # then d_v <- d_u + 1
            d[v] = new_d
            # enqueue(Q, v)
            enqueue(q, v)
    # return d
    return list(d.values())


def distanceDistribution(G: GraphData):
    def BFS_reducer(l: list, s: int):
        print(s) if s % 100 == 0 else None
        return l + BFS(G, s)
    distances = list(filter(lambda x: x != 0, reduce(BFS_reducer, G.keys(), [])))
    df = pd.DataFrame(Counter(distances).items(), columns=['Steps', 'Count']).sort_values('Steps')
    df['Percent'] = df['Count'].apply(lambda x: x/len(distances) * 100)
    return df


def test_code():
    start = datetime.now()
    distribution = distanceDistribution(loadGraph())
    print(distribution)
    print(datetime.now() - start)
    return distribution


"""
Despite several thousands of vertices with millions of edge traversals, no single node is more than
8 edges away from any other.

In fact, there is a ~98% probability that a given node can traverse to any other node by traveling along 6 or less edges. 
This holds true to the "6 degrees of separation" SWP mantra.

Steps    Count    Percent
    1   176468   1.081996
    2  2716134  16.653711
    3  3981852  24.414338
    4  5861560  35.939584
    5  2565170  15.728090
    6   677214   4.152272
    7   315464   1.934237
    8    15620   0.095773

The data appears normally distributed, with a right tail. It would be interested to see how the mean and distribution
is effected by increasing population size. Does the mean shift as a function of population size? Is it logarithmic? These
would be interesting questions to explore further.


"""
