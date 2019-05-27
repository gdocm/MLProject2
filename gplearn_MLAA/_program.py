"""The underlying data structure used in gplearn_MLAA.

The :mod:`gplearn_MLAA._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn_MLAA.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement
from scipy.spatial.distance import euclidean
from .functions import _Function, _function_map, sig1, tanh1
from .utils import check_random_state
from .Node import Node

class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn_MLAA.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 semantical_computation=False,
                 library = None):
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.semantical_computation = semantical_computation
        self.library = library

        if self.program is None:
            # Create a naive random program
            self.program = self.build_program(random_state)
            if self.semantical_computation:
                self.program_length = None
                self.program_depth = None
        elif self.semantical_computation:
            self.program_length = self.program[1]
            self.program_depth = self.program[2]
            self.program = self.program[0]
        else:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        #self.treeTransform()
        
    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        if self.semantical_computation and self.program_depth is not None:
            return self.program_depth
        else:
            terminals = [0]
            depth = 1
            for node in self.program:
                if isinstance(node, _Function):
                    terminals.append(node.arity)
                    depth = max(len(terminals), depth)
                else:
                    terminals[-1] -= 1
                    while terminals[-1] == 0:
                        terminals.pop()
                        terminals[-1] -= 1
            return depth - 1

    def _depth_program(self, program):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        if self.semantical_computation and self.program_length is not None:
            return self.program_length
        else:
            return len(self.program)

    def execute(self, X, backP= False):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int) or isinstance(node, np.int32):
            return X[:, node]

        apply_stack = []
        if backP:
            semantics = []
        for ix,node in enumerate(self.program):
            if isinstance(node, _Function):
                apply_stack.append([(node, ix)])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0][0]
                i = apply_stack[-1][0][1]
                
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int) or isinstance(t, np.int32)
                             else t for t in apply_stack[-1][1:]]
                
                intermediate_result = function(*terminals)

                if backP:
                    
                    semantics.append((intermediate_result,i))
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    if backP:
                        return semantics
                    return intermediate_result

        # We should never get here
        return None

    @staticmethod
    def execute_(program, X):
        """Execute the program according to X.

        Parameters
        ----------
        program : list
            The flattened tree representation of the program.

        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = program[0]
        
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int) or isinstance(node, np.int32):
            return X[:, node]
        apply_stack = []
        for node in program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int) or isinstance(t, np.int32)
                             else t for t in apply_stack[-1][1:]]
                
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def raw_fitness_semantics(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness, y_pred

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None, depth_probs = False):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        
        
        #If probs depends on depth
        if depth_probs:
            depths = self.pointDepth(program)
            probs = np.array([probs[i]*(1/np.max(depths))*(depths[i]+1) for i in range(len(probs))])
        
        probs = np.cumsum(probs / probs.sum())
            
        start = np.searchsorted(probs, random_state.uniform())
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state, depth_probs = False):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state, depth_probs = depth_probs)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor, depth_probs = depth_probs)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] + donor[donor_start:donor_end] + self.program[end:]), removed, donor_removed
    
    def treeTransform(self):
        struct = np.array(self.program.copy())
        
        if not isinstance(struct[0], _Function):
            return
        
        for i in range(len(struct)-1, -1,-1):
            node = self.program[i]
            
            if isinstance(node, _Function):
                children = struct[i+1:i+1+node.arity]
                temp = Node(node, children = children)
                struct = np.delete(struct,range(i+1, i+1+node.arity))
                struct[i] = temp
                
        struct[0].setDepth()
        return struct[0]
    
    def pointDepth(self, program):
        depths = [0]
        depth = 1
        arities = [2]
        for i in range(1, len(program)):
            node = program[i]
            
            if isinstance(node, _Function):
                arities[-1] -= 1
                arities.append(node.arity)
                depths.append(depth)
                depth += 1
            else:
                arities[-1] -= 1
                depths.append(depth)
            while arities[-1] == 0:
                depth -= 1
                del arities[-1]
                if len(arities) == 0:
                    break
            
            
        return depths
    
    def gs_crossover(self, donor, random_state):
        """Perform the Geometric Semantic crossover operation on the program."""
        random_tree = [sig1] + self.build_program(random_state)
        return [_function_map['add'], _function_map['mul']] + random_tree + \
            self.program + [_function_map['mul'], _function_map['sub'], 1.0] + random_tree + donor

    def gs_crossover_semantics(self, X, donor, random_state):
        """Perform the Geometric Semantic crossover operation on the program."""
        rt = [sig1] + self.build_program(random_state)
        rt_semantics = _Program.execute_(rt, X)
        offspring_length = self.program_length + donor.program_length + len(rt)*2 + 5
        offspring_depth = np.maximum(np.maximum(self.program_depth, donor.program_depth)+2, self._depth_program(rt)+3)
        return (np.add(np.multiply(self.program, rt_semantics), np.multiply(np.subtract(1, rt_semantics), donor.program)),
                offspring_length, offspring_depth)
    
    def grasm_mutation(self, X, random_state, alpha = 0.5, depth_probs=False):
        start, end = self.get_subtree(random_state, depth_probs = depth_probs)
        p = self.program[start:end]
        p_semantics = _Program.execute_(p,X)
        distances = [euclidean(prgs[1], p_semantics) for prgs in self.library]
        d_min = np.min(distances)
        d_max = np.max(distances)
        threshold = d_min + alpha*(d_max - d_min)
        rcl = np.array(self.library)[np.array(distances) <= threshold]
        p = rcl[np.random.choice(len(rcl))][0]
        return self.program[:start] + p + self.program[end:]
    
    def gs_mutation_sig(self, ms, random_state):
        """Perform the Geometric Semantic operation on the program."""
        rt1 = [sig1] + self.build_program(random_state)
        rt2 = [sig1] + self.build_program(random_state)
        return [_function_map['add']] + self.program + [_function_map['mul'], ms, _function_map['sub']] + rt1 + rt2

    def gs_mutation_tanh(self, ms, random_state):
        """Perform the Geometric Semantic operation on the program."""
        random_tree = [tanh1] + self.build_program(random_state)
        return [_function_map['add']] + self.program + [_function_map['mul'], ms] + random_tree

    def gs_mutation_tanh_semantics(self, X, ms, random_state):
        """Perform the Geometric Semantic operation on the program."""
        random_tree = [_function_map['mul'], ms, tanh1] + self.build_program(random_state)
        offspring_length = self.program_length + len(random_tree) + 1
        offspring_depth = 1 + np.maximum(self.program_depth, self._depth_program(random_tree))
        return (np.add(self.program, _Program.execute_(random_tree, X)), offspring_length, offspring_depth)

    def subtree_mutation(self, random_state, depth_probs = False):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state, depth_probs)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)
    
    def competent_mutation(self, X, y, oracle, random_state, depth_probs = False):
        coords = self.programCoords()
        old_semantics = self.execute(X, True)
        start, end = self.get_subtree(random_state, depth_probs = depth_probs)
        node = (self.program[start], coords[start])
        new_semantics = self.semanticBackPropagation(y, node, X, coords, old_semantics)
        node = oracle(new_semantics)
        return self.program[:start] + node + self.program[end:]

    def semanticBackPropagation(self, D, target, X, coords, old_semantics):
        '''
        Calculate desired semantics for target node
        '''
        #print("Performing semantic backpropagation")
        node = (self.program[self.nodeByArity(target[1][0], coords)],[target[1][0]])
        for s in range(len(D)): #For semantic in target semantics
            s_ = D[s]
            e = 0
            while node != target:
                e += 1


                
                ni = self.nextNode(node[1],target,coords)
                x = (self.program[ni], coords[ni])
                
                #Set arities for the other node
                if x[1][-1] == 1:
                    temp = x[1].copy()
                    temp[-1] = 0
                elif x[1][-1] == 0:
                    temp = x[1].copy()
                    temp[-1] = 1
                    
                #Get semantics of the other node
                other_i = self.nodeByArity(temp,coords)
                other = self.program[other_i]
                if isinstance(other, _Function):
                    other = list(filter(lambda x: x[1] == other_i, old_semantics))[0][0]
                    if not isinstance(other, int) and not isinstance(other, np.int32) and not isinstance(other, float):
                        other = other[s]
                elif isinstance(other,int) or isinstance(other, np.int32):
                    other = X[s,other]

                #Set position of sub-target node
                k = x[1][-1]
                
                #Get new semantics of sub-target node
                s_ = node[0].invert(k, s_, other)
                node = x
                
                #Get Next Node
                if e > 100:
                    print("Program",self.program)
                    print("Current Node:",node)
                    print("Target Node:",target)
                    raise(TypeError,"Stuck in loop")

            D[s] = s_
        return D
                
    def nextNode(self, node, target, coords):
        '''
        Returns the next node on the path to the target
        '''
        
        n_coords = target[1]
        if n_coords[-1] == 2:
            del n_coords[-1]
        if node[-1] == 2:
            del node[-1]
        
        n_coords = list(np.array(n_coords)[:len(node)+1])
        return self.nodeByArity(n_coords, coords)
    
    def nodeByArity(self,node_arity, coords):
        '''
        Returns index of node by arity list
        '''
        indexes = list(range(len(coords)))
        coords = [tuple(coord) for coord in coords]
        
        dict_ = dict(zip(coords,indexes))
        if isinstance(node_arity, int) or isinstance(node_arity, np.int32):
            node_arity = [node_arity]
        return dict_[tuple(node_arity)]
    

    
    def programCoords(self):
        arities = [-1,2]
        coords = [[-1]]
        for i in range(1, len(self.program)):
            node = self.program[i]
            if isinstance(node, _Function):
                arities[-1] -= 1
                arities.append(node.arity)
            else:
                arities[-1] -= 1
            if arities[-1] == 2:
                temp = arities.copy()
                del temp[-1]
                coords.append(temp.copy())
            else:
                coords.append(arities.copy())
            while arities[-1] == 0:
                del arities[-1]
                if len(arities) == 0:
                    break
        #print("Coords", coords)
        return coords
            
                    

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)