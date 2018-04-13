import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import deque

class Contagion:
    """
    The Contagion class is a Monte-Carlo simulator that allows to calculate infection arrival times for SIR processes on complex networks.
    We assume a standard contact process, i.e. an infected individual transmits the pathogen with a uniform infection rate to a susceptible
    neighbor and recovers with a (uniform) recovery rate. The continuous time problem is mapped to a percolation problem as descibed in 
    ```
    Kenah E., Robins J.M., "Second look at the spread of epidemics on networks", Phys. Rev. E (2007)
    ```
    This approach leads to an efficient implementation because every realization, i.e. every Monte-Carlo step, returns the results for all outbreak locations at once. 
    """
    def __init__(self, graph=None, verbose=True, **kwargs):
        self.graph = None
        self.Nnodes = 0
        self.Nedges = 0
        self.infection_rate = 0.
        self.recovery_rate = None
        self.ensemble = None
        self.rel_error = None
        self.max_ensemble = None
        self.infection_arrival_matrix = None
        self.infection_count_matrix = None
        self.error = None
        self.arrival_statistics = None
        
        if graph is not None:
            self.load(graph, verbose=verbose, **kwargs)
    
    def __str__():
        txt = "The Contagion class is a Monte-Carlo simulator that allows to calculate infection arrival times for SIR processes on complex networks. We assume a standard contact process, i.e. an infected individual transmits the pathogen with a uniform infection rate to a susceptible neighbor and recovers with a (uniform) recovery rate. The continuous time problem is mapped to a percolation problem. This approach leads to an efficient implementation because every realization, i.e. every Monte-Carlo step, returns the results for all outbreak locations at once."
    
        print txt,"\n--------------------------------------"
        print "graph:", self.graph
        print "infection rate:", self.infection_rate
        print "recovery rate:", self.recovery_rate
    
    def load(self, graph, verbose=True, directed=True, strongly_connected=False, **kwargs):
        """
        Load an edgelist from a file (str), a Numpy array, list or take a networkx graph. The edgelist is supposed to have two or three columns: source, target, weight.
        If only two columns are provided, we assume a uniform edge weight. Node IDs have to run continuously from 0 to Number_of_nodes-1 and the final graph is saved internally in self.graph.
        Parameters
        ----------
            graph : str, numpy.array, list or networkx.DiGraph / networkx.Graph
                If type(graph) == str, assume a path to the edgelist file readable 
                by numpy.genfromtxt. Additional keywords such as "dtype", "delimiter" and "comments"
                can be past to the function.
                If type(graph) == numpy.array, assume a two/three column edgelist
                If type(graph) == list, assume an edgelist, convertible via np.array(graph)
                If type(graph) == networkx.Graph the network will be converted to a DiGraph
                weights are stored as edge attributes "weight".
            
            verbose : bool
                Print information about the data. True by Default
            
            directed : bool
                Interprete the edgelist accordingly (True by default). If False, then reciprical edges are added.
                In any case the resulting graph is directed.  

            strongly_connected : bool
                If True (default), then only the giant strongly connected component will be considered.
                
            **kwargs
                Default parameters are passed to numpy.genfromtxt. Some of the supported keywords are
                    dtype     : float (default)
                    delimiter : ","   (default)
                    comments  : "#"   (default)
        """
        if isinstance(graph, str): #expect path to file with edgelist
            edgelist = np.genfromtxt(graph, unpack=False, **kwargs)
            if edgelist.shape[1] == 2:
                graph = np.hstack((edgelist,np.ones((edgelist.shape[0],1))))
            elif edgelist.shape[1] != 3:
                raise ValueError("Expected a two or three column edgelist. Instead got {} columns".format(edgelist.shape[1]))
            if verbose:
                print "file successfully loaded..."
            graph = edgelist

        if isinstance(graph, np.ndarray) or isinstance(graph, list):
            edgelist = np.array(graph)
            source, target = edgelist[:,0].astype(int), edgelist[:,1].astype(int) #source and target ID as int!
            if edgelist.shape[1] == 2:
                flux = np.ones_like(edgelist[:,1], dtype=float)
            elif edgelist.shape[1] == 3:
                flux = edgelist[:,2].astype(float)
            else:
                raise ValueError("Expected a two or three column edgelist. Instead got {} columns".format(edgelist.shape[1]))
            nodes  = set(source) | set(target)
            if set(xrange(len(nodes))) != nodes:
                new_node_ID = {old:new for new,old in enumerate(nodes)}
                map_new_node_ID = np.vectorize(new_node_ID.__getitem__)
                source = map_new_node_ID(source)
                target = map_new_node_ID(target)
                if verbose:
                    print "\nThe node IDs have to run continuously from 0 to Number_of_nodes-1."
                    print "Node IDs have been changed according to the requirement.\n-----------------------------------\n"
            self.nodes = len(nodes)
            
            graph = nx.DiGraph()
            graph.add_weighted_edges_from(zip(source, target, flux), weight="weight")
        
        if isinstance(graph, nx.Graph):
            graph = graph.to_directed()

        if isinstance(graph, nx.DiGraph):
            for edge in graph.edges(data=True):
                if not "weight" in edge[2]:
                    weight_dct = {e: 1. for e in graph.edges()} #map edge ID to infection time
                    nx.set_edge_attributes(graph, weight_dct, name="weight") 
                    if verbose:
                        print "Missing edge attribute 'weight'. Added unit value..."
                break

            if graph.number_of_selfloops() > 0:
                graph.remove_edges_from( graph.selfloop_edges() )
                if verbose:
                    print "Self loops detected and removed..."
            
            if not directed:
                if nx.algorithms.reciprocity(graph) != 1:
                    graph.add_weighted_edges_from(zip(target, source, flux), weight="weight")
                    if verbose:
                        print "reciprical edges added..."

            if strongly_connected:
                if not nx.is_strongly_connected(graph):
                    ccs = nx.strongly_connected_component_subgraphs(graph)
                    GSCC = sorted(ccs, key=len, reverse=True)[0]
                    if verbose:
                        print "Selected the giant strongly connected component..."
                        print "nodes and edges before:", graph.number_of_nodes(), graph.number_of_edges()
                        print "nodes and edges now:", GSCC.number_of_nodes(), GSCC.number_of_edges()
                    if GSCC.number_of_nodes() != graph.number_of_nodes():
                        if verbose:
                            graph = nx.convert_node_labels_to_integers(GSCC)
                            print "Disconnected nodes have been removed and the remaining IDs have been relabeled..."
            
            self.graph = graph
            self.Nnodes = graph.number_of_nodes()
            self.Nedges = graph.number_of_edges()
            if verbose:
                print "graph has been saved in self.graph"
                print "number of nodes and edges:", self.Nnodes, self.Nedges
        
        else:
            raise ValueError("Expected type of graph is either str, numpy.ndarray or nx.DiGraph. Instead got {}".format(type(graph)))
    
    ##############################################################################
    def contagion(self, 
                                infection_rate=1.,
                                recovery_rate=None,
                                ensemble=1,
                                verbose=True,
                                statistics=False,
                                source = None,
                                rtol=0):
        """
        Monte-Carlo simulator for infection arrival times of a SIR process on complex, static networks.
        We assume a standard contact process, i.e. an infected individual transmits the pathogen with a uniform infection rate to a susceptible
        neighbor and recovers with a (uniform) recovery rate. The continuous time problem is mapped to a percolation problem as descibed in 
        ```
        Kenah E., Robins J.M., "Second look at the spread of epidemics on networks", Phys. Rev. E (2007)
        ```
        This approach leads to an efficient implementation because every realization, i.e. every Monte-Carlo step, returns the results for all outbreak locations at once.
        The edge weight parameter multiplies with the infection rate thus allowing for heterogenous transmission rates. Non-Markovian infection rates and heterogenous recovery rates are not yet implemented but, can be added if required.

        The infection arrival times are added up and saved internally as the numpy array self.infection_arrival_matrix.
        We also count the number of times a node has been infected and store the result in the numpy array self.infection_arrival_matrix.
        If no outbreak location is given, these arrays are of size (number of nodes, number of nodes) and indexed as (source, target).
        Otherwise the array is of size (number of nodes,) and indexed as (target,).
        The relative array is calculated every 100 steps and is stored as the numpy array self.error

        Parameter
        ------------
            infection_rate : float
            
            recovery_rate : float,
            
            ensemble : int
                Maximum number of Monte-Carlo steps. If the relative error falls below "rtol", then the simulation breaks and the current number of Monte-Carlo steps is saved in self.ensemble
            
            verbose : bool
                Print progress information
            
            statistics : bool
                Save all infection arrival times internally as self.arrival_statistics (False by default). 
                If source = None the data is stored in a multidimensional numpy array (source, target, arrival times) of size (nodes, nodes, ensemble).
                Otherwise the data is stored in an array of size (nodes, ensemble) as (target, arrival times).
                Beware, this option may require a lot of storage depending on the ensemble size.

            source : int or tuple
                If source is None, all nodes are considered as outbreak locations and the 
                If source is in self.graph.nodes(), then the outbreak location is fixed.
                Otherwise source can also be a n-tuple where the elements are considered multiple outbreak locations.
                In this case the infection arrival time refers to the first transmission from any outbreak location to the target node. 
            
            rtol : float
                The relative error is calculated every 100 Monte-Carlo steps for all mean infection arrival times. The largest deviation is added to a list of 10 elements and the largest element of the list will then be compared to rtol to minize fluctuations. You can choose 0 <= rtol < 1. If the largest relative error is smaller than rtol, the simulation breaks and the current ensemble size is saved in self.ensemble.
        """
        self.recovery_rate = recovery_rate
        self.infection_rate = infection_rate
        self.ensemble = int(ensemble)
        self.rtol = rtol
        
        Nnodes = self.Nnodes
        error_step = 100 #calculate the relative error only after 'error_step' Monte-Carlo steps
        deque_length = 10 #save the relative error in a list of constant size 
        error_deque = deque([1e5]*deque_length, deque_length) # a new element replaces the oldest in a deque
        error = np.empty(int(ensemble/error_step)) #save the error trajectory here

        if source is None:
            shortest_path_length = nx.shortest_path_length
            infection_arrival_matrix = np.zeros((Nnodes,Nnodes)) #add infection arrival times here
            infection_count_matrix = np.zeros((Nnodes,Nnodes)) #count successfull transmissions
            old = np.ones((Nnodes,Nnodes))*1e6 #save old arrival times for error estimation
            if statistics:
                arrival_statistics = np.empty((Nnodes, Nnodes, ensemble))*np.nan #arrival statistics for one predefined source node
            multiple_source = False

        elif source in self.graph.nodes():
            shortest_path_length = nx.shortest_path_length
            infection_arrival_matrix = np.zeros(Nnodes) #add infection arrival times here
            infection_count_matrix = np.zeros(Nnodes) #count successfull transmissions
            old = np.ones(Nnodes)*1e6 #save old arrival times for error estimation
            if statistics:
                arrival_statistics = np.empty((Nnodes, ensemble))*np.nan #arrival statistics for one predefined source node
            multiple_source = False

        elif isinstance(source, tuple):
            assert all([n in self.graph.nodes() for n in source]), "all source IDs need to be in self.graph.nodes()"
            shortest_path_length = nx.multi_source_dijkstra_path_length
            infection_arrival_matrix = np.zeros(Nnodes)#add infection arrival times here
            infection_count_matrix = np.zeros(Nnodes) #count successfull transmissions
            old = np.ones(Nnodes)*1e6 #save old arrival times for error estimation
            if statistics:
                arrival_statistics = np.empty((Nnodes, ensemble))*np.nan #arrival statistics for one predefined source node
            assert all([n in self.graph.nodes() for n in source]), "all source IDs need to be in self.graph.nodes()"
            multiple_sources = True
        else:
            raise ValueError("'source' is not chosen appropriately")

        if verbose:
            ensemble_iter = tqdm(xrange(ensemble))
        else:
            ensemble_iter = xrange(ensemble)
        
        for ii in ensemble_iter:
            
            if not recovery_rate:
                self.get_SI_transmission_graph() #add transmission times as edge weights 
                tg = self.graph #original graph equals transmission graph 
            else:# For SIR dynamics the tg is a subgraph with the same number of nodes
                tg = self.get_SIR_transmission_graph()# add also transmission times
                        
            if source is None:
                time_iter = nx.shortest_path_length(tg, weight="time") #shortest path gives infection arrival time
                for s, time_dct in time_iter: #s, t -> source, target
                    for t in time_dct:
                        infection_arrival_matrix[s, t] += time_dct[t]
                        infection_count_matrix[s, t] += 1.
                        if statistics:
                            arrival_statistics[s, t, ii] = time_dct[t]
            
            else:
                time_dct = shortest_path_length(tg, source, weight="time") 
                for t in time_dct: #t -> target
                    infection_arrival_matrix[t] += time_dct[t]
                    infection_count_matrix[t] += 1.
                    if statistics:
                        arrival_statistics[t, ii] = time_dct[t]
            
            if ( ii % error_step == 0 ) and ii > 0:
                error_deque.append( np.max(np.abs(  (infection_arrival_matrix/float(ii) - old)/old  )) )
                max_error = max(error_deque)
                error[int(ii/error_step)] = max_error
                old = np.ma.masked_equal(infection_arrival_matrix/float(ii),0)
                if max_error < rtol:
                    pass#break

        self.infection_arrival_matrix = infection_arrival_matrix
        self.infection_count_matrix = infection_count_matrix
        self.error = error
        self.ensemble = ii + 1
        if statistics:
            self.arrival_statistics = arrival_statistics

    ###############################################################################

    def get_SIR_transmission_graph(self):
        infection_rate = self.infection_rate

        recovery_time = {n: rec for n, rec in zip(xrange(self.Nnodes), np.random.exponential(scale=1./self.recovery_rate, size=self.Nnodes))}
        tg = nx.DiGraph() #transmission graph (tg)
        tg.add_nodes_from(self.graph.nodes())

        rnd = np.random.random(self.Nedges) #calculate random numbers in advance
        cnt = 0
        for n, nbrsdict in self.graph.adjacency(): #iterate through all edges: n->node and nbrsdict->dictionary with nodes and edge attributes
            for nbr, eattr in nbrsdict.items(): #nbr->node (out-neighbor) and eattr->edge attribute of edge(n,nbr)
                avg_infection_time  = 1. / infection_rate / eattr["weight"] #we assume that the transmission rate scales with the edge weight
                transmission_prob = 1. - np.exp( - recovery_time[n] / avg_infection_time) #determine the transmission probability for one edge
                if transmission_prob > rnd[cnt]: #create one realisation of the transmission graph
                    infection_time = np.random.exponential(scale=avg_infection_time) #determine the infection time
                    tg.add_edge(n, nbr, time=infection_time) #add edge to the transmission graph 
                cnt += 1
        return tg
    
    def get_SI_transmission_graph(self):
        infection_rate = self.infection_rate
        for n, nbrsdict in self.graph.adjacency(): #iterate through all edges: n->node and nbrsdict->dictionary with nodes and edge attributes
            for nbr, eattr in nbrsdict.items(): #nbr->node (out-neighbor) and eattr->edge attribute of edge(n,nbr)
                avg_infection_time  = 1. / infection_rate / eattr["weight"] #we assume that the transmission rate scales with the edge weight
                self.graph[n][nbr]["time"] = np.random.exponential(scale=avg_infection_time) #determine the infection time
