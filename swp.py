"""
Christopher Kramer

Despite several thousands of vertices with millions of edge traversals, no single node is more than
8 edges away from any other.

In fact, there is a ~98% probability that a given node can traverse to any other node by traveling along 6
or less edges. This holds true to the "6 degrees of separation" SWP mantra.

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
is effected by increasing population size. Does the mean shift as a function of population size? Is it logarithmic?
These would be interesting questions to explore further.

"""

import warnings
from collections import defaultdict, deque, Counter
from functools import reduce
from itertools import repeat
from operator import eq, gt
from typing import List, Optional, NoReturn, Any, Type, Iterable, Deque, DefaultDict, Set, Sized
from datetime import datetime

import pandas as pd

# typehints
FilePath = str
Vertex = int
Distances = List
DataElement = Any
DataElements = Deque
AdjacencyDefaultDict = DefaultDict[Vertex, Set[Vertex]]
GraphData = AdjacencyDefaultDict
UsageWarning = NoReturn


class MyQueue:
    """
    Uses a collections.deque as the underlying data structure
    """

    def __init__(self, dtype: Type):
        """
        MyQueue custom class. Has an inherent datatype, type checking, enqueueing and dequeueing.
        :param dtype:
        """
        self.__data = deque()
        self.__type = dtype

    @property
    def data(self) -> DataElements: return self.__data  # Returns data in queue without popping

    # noinspection PyUnusedLocal
    @data.setter
    def data(self, elements: Iterable) -> UsageWarning:
        """
        Do not use - cannot overwrite pre-queued elements with new iterable.
        :param elements:
        :return:
        """
        warnings.warn("Cannot replace existing queue data with a static structure. \
         Please create a new MyQueue object instead or enqueue \
        elements using self.enqueue.")

    def __data_type_check(self, element) -> Optional[bool]:
        """
        Checks if element matches queue built-in dtype
        :param element:
        :return:
        """
        return True if eq(self.__type, type(element)) else warnings.warn("Element dtype does not match queue dtype")

    def enqueue(self, element: DataElement) -> NoReturn:
        """
        Enqueues element using deque.append()
        :param element:
        :return:
        """
        self.__data.append(element) if self.__data_type_check(element) else None

    def dequeue(self) -> DataElement:
        """
        Dequeues an element using deque.popleft()
        :return:
        """
        return self.__data.popleft()

    def front(self) -> DataElement:
        """
        Peeks at first element of MyQueue without dequeueing
        :return:
        """
        return self.__data[0]

    def queue(self) -> bool:
        """
        Checks status of queue
        :return:
        """
        return gt(len(self.data), 0)


def enqueue(q: MyQueue, s: Vertex) -> NoReturn: q.enqueue(s)  # enqueue utility function to match pseudocode


def loadGraph(edge_file_path: FilePath = 'edges.txt') -> GraphData:
    """
    Returns a default dictionary of vertex : [neighbors] pairs
    :param edge_file_path:
    :return:
    """

    def create_edge_list(raw_data: str) -> list:
        """
        Creates a list of [A, B] integer pairs representing node connections
        :param raw_data:
        :return:
        """

        def str_splitter(string: str) -> list: return string.split(' ')

        def length_checker(l: Sized, length: int = 2) -> bool: return eq(len(l), length)

        def str_list_to_int_list(l: list) -> list: return list(map(lambda x: int(x), l))

        return list(
            # convert str list to int list
            map(str_list_to_int_list,
                # gets rid of non-paired vertex neighbors (EOF usually)
                filter(length_checker,
                       # split vertex pairs from file
                       map(str_splitter, raw_data.split('\n')))))

    def create_adjacency_list_from_edge_list(empty_adj_list: AdjacencyDefaultDict,
                                             edge_list: list) -> AdjacencyDefaultDict:
        def update_adjacency_list(d: AdjacencyDefaultDict, edge: list) -> NoReturn:
            # Updates vertex : neighbor pairs and the complement (neighbor : vertex)
            d[edge[0]].add(edge[1])
            d[edge[1]].add(edge[0])

        # Converts node connection from type list to adjacency dictionary (node: connections)
        deque(map(update_adjacency_list, repeat(empty_adj_list), edge_list), 0)
        return empty_adj_list

    # Open Edge file
    with open(edge_file_path, 'r') as fp:
        raw_edges: list = create_edge_list(fp.read())

    # Convert to adj. list & return
    return create_adjacency_list_from_edge_list(empty_adj_list=defaultdict(lambda: set()),
                                                edge_list=raw_edges)


def BFS(G: GraphData, s: Vertex) -> Distances:
    """
    BFS function following breadth-first-search algo
    :param G:
    :param s:
    :return:
    """
    # Initialize an empty queue Q
    q = MyQueue(int)
    # for each u \in V do d_u <- \infty
    d: DefaultDict[int] = defaultdict(lambda: None)
    # ds <- 0
    d[s] = 0
    # Enqueue(Q, s)
    enqueue(q, s)
    # while Q is not empty
    while q.queue():
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


def distanceDistribution(G: GraphData) -> pd.DataFrame:
    """
    Constructs a
    :param G:
    :return:
    """

    # Reduction function to run BFS iteratively: runs BFS, appends to master list
    def bfs_reducer(l: list, s: int) -> list:
        if s % 100 == 0:
            print(s)
        return l + BFS(G, s)

    # Creates the list containing iterative BFS searches over the vertexes in G except the start vertex (x != 0)
    distances: list = list(filter(lambda x: x != 0, reduce(bfs_reducer, G.keys(), [])))
    # Grouping step distances using Counter
    df: pd.DataFrame = pd.DataFrame(Counter(distances).items(), columns=['Steps', 'Count']).sort_values('Steps')
    # Converting counts of steps to percentages
    df['Percent'] = df['Count'].apply(lambda x: x / len(distances) * 100)
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

In fact, there is a ~98% probability that a given node can traverse to any other node by traveling along 6 
or less edges. This holds true to the "6 degrees of separation" SWP mantra.

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
is effected by increasing population size. Does the mean shift as a function of population size? Is it logarithmic? 
These would be interesting questions to explore further.


"""
