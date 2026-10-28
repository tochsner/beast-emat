package emat.stochasticmapping;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.util.Randomizer;
import emat.helper.NodeStateLookup;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Samples the complete history of a single site on the tree, or on a region of it, from its
 * exact conditional distribution under the model of the genetic prior, given the tip data
 * (stochastic mapping). Per site, the genetic prior is the density of a CTMC path on the
 * tree, so a draw consists of sampling the node states by pruning and then the path on every
 * branch given its end states. The node states are always sampled exactly, while the paths on
 * the branches are sampled by the given stochastic mapping, which may approximate the model.
 * <p>
 * A region is a top node and the free nodes below it, together with the branches above them,
 * down to the boundary nodes, whose states stay fixed like those of observed tips. The
 * branches above the boundary nodes belong to the region. The state of the top follows the
 * transition probabilities from the fixed state of its parent, or the equilibrium frequencies
 * at the root, as below a branch of infinite length. The whole tree is the region with the
 * root at the top and no boundary. A region may also hold a clade below one of its nodes
 * whose tips have no data at the sampled sites: such a clade does not inform the states
 * above it, so it is left out of the pruning, and its history is simulated forwards from the
 * sampled state above it, at a cost per site that follows the number of its mutations
 * rather than its size.
 */
public class SiteHistorySampler {

    // marks a node whose partial likelihoods allow every state equally, like a tip whose data is missing
    private static final int UNINFORMATIVE = -2;

    final Mutations mutations;
    final SubstitutionProcess process;
    final TreeInterface tree;
    final Alignment alignment;
    final DataType dataType;
    final StochasticMapping stochasticMapping;

    final int numStates;

    // the rate matrix Q that the stochastic mapping and the cached transition probabilities were computed for
    double[] rateMatrix;

    // the pruned nodes of the current region in post-order, so that every child precedes its parent and the top
    // comes last, and the stack used to collect them
    final Node[] regionNodes;
    final Node[] traversalStack;
    int numRegionNodes;

    // the clade of the current region without data, which may be null, its nodes in pre-order, and the starts of
    // the branches above them when laid end to end in scaled time
    Node dataFreeClade;
    final Node[] cladeNodes;
    final double[] cladeBranchStarts;
    int numCladeNodes;

    // per node: the state after the last simulated mutation on the branch above it, valid if stamped with the
    // current simulation
    final int[] simulatedEndStates;
    final long[] simulatedEndStamps;
    long simulationStamp;

    // the boundary nodes of the region with the lookups of their states, and the lookup of the states at the
    // parent of the top, which is unused at the root. A node is on the boundary if its flag is stamped with the
    // current region
    final List<Node> boundaryNodes = new ArrayList<>();
    final List<NodeStateLookup> boundaryStates = new ArrayList<>();
    final NodeStateLookup topParentStates;
    final long[] boundaryStamps;
    long regionStamp;

    // per tip node: the index of its taxon in the alignment
    final int[] taxonIndices;

    // per character code: the tip partial likelihoods, and the only state compatible with
    // it, -1 if there are several, or UNINFORMATIVE if all are, created when first needed
    double[][] tipPartials = new double[0][];
    int[] tipStates = new int[0];

    // per node: the scale of Q on the branch above it, the scaled branch length for which
    // the transition probabilities were computed, the transition probabilities of the
    // branch, the partial likelihoods of the subtree below it, the only state its partial
    // likelihoods allow, -1 if they allow several, or UNINFORMATIVE if they allow all equally,
    // and its sampled state
    final double[] rateScales;
    final double[] durations;
    final double[][] transitionProbabilities;
    final double[][] partials;
    final int[] knownStates;
    final int[] nodeStates;

    // reusable weights of the states of a node, from which its state is drawn
    final double[] stateWeights;

    public SiteHistorySampler(Mutations mutations, SubstitutionProcess process, StochasticMapping stochasticMapping) {
        this.mutations = mutations;
        this.process = process;
        this.tree = mutations.getTree();
        this.alignment = mutations.getAlignment();
        this.dataType = this.alignment.getDataType();

        this.numStates = this.alignment.getMaxStateCount();
        this.stochasticMapping = stochasticMapping;

        int numNodes = this.tree.getNodeCount();
        int numSites = mutations.getReferenceSequence().length;
        this.regionNodes = new Node[numNodes];
        this.traversalStack = new Node[numNodes];
        this.cladeNodes = new Node[numNodes];
        this.cladeBranchStarts = new double[numNodes + 1];
        this.simulatedEndStates = new int[numNodes];
        this.simulatedEndStamps = new long[numNodes];
        this.topParentStates = new NodeStateLookup(mutations, numSites);
        this.boundaryStamps = new long[numNodes];
        this.rateScales = new double[numNodes];
        this.durations = new double[numNodes];
        this.transitionProbabilities = new double[numNodes][this.numStates * this.numStates];
        this.partials = new double[numNodes][this.numStates];
        this.knownStates = new int[numNodes];
        this.nodeStates = new int[numNodes];
        this.stateWeights = new double[this.numStates];

        Arrays.fill(this.durations, Double.NaN);

        // the tips keep their numbers and taxa when the tree changes

        Map<String, Integer> taxonIndicesByName = new HashMap<>();
        List<String> taxaNames = this.alignment.getTaxaNames();
        for (int taxonNr = 0; taxonNr < taxaNames.size(); taxonNr++) {
            taxonIndicesByName.put(taxaNames.get(taxonNr), taxonNr);
        }

        this.taxonIndices = new int[numNodes];
        for (Node tip : this.tree.getExternalNodes()) {
            this.taxonIndices[tip.getNr()] = taxonIndicesByName.get(tip.getID());
        }
    }

    /**
     * Takes the current model from the substitution process and sets the region to the whole
     * tree. This has to be called again whenever the tree or the model changes, and before
     * sampleSiteHistory and computeLogSiteHistoryDensity.
     */
    public void updateModel() {
        this.setUpRegion(this.tree.getRoot(), node -> true, null);
    }

    /**
     * Takes the current model from the substitution process and sets the region to the given
     * top and the nodes below it that are free, down to the first nodes that are not, which
     * form the boundary. Whether a node is free is only asked for the children of free nodes and
     * of the top. The transition probabilities of a branch of the region are only recomputed if
     * Q or the scaled length of the branch has changed. The states of the boundary and of the
     * parent of the top are read from the history when a site is first sampled, so the history
     * must not change in between. The data-free clade, which may be null, is a child of a node
     * of the region whose tips must have no data at the sampled sites. It is not asked whether
     * it is free, and it belongs to the region with all nodes below it.
     */
    public void setUpRegion(Node top, Predicate<Node> isFree, Node dataFreeClade) {
        this.process.update();
        if (this.process.getRateMatrix() != this.rateMatrix) {
            this.rateMatrix = this.process.getRateMatrix();
            this.stochasticMapping.setRateMatrix(this.rateMatrix);
            Arrays.fill(this.durations, Double.NaN);
        }

        this.regionStamp++;
        this.boundaryNodes.clear();
        this.dataFreeClade = dataFreeClade;
        this.computeRegionPostOrder(top, isFree);

        while (this.boundaryStates.size() < this.boundaryNodes.size()) {
            this.boundaryStates.add(new NodeStateLookup(this.mutations, this.mutations.getReferenceSequence().length));
        }
        for (int i = 0; i < this.boundaryNodes.size(); i++) {
            this.boundaryStates.get(i).resetFor(this.boundaryNodes.get(i));
        }
        if (!top.isRoot()) {
            this.topParentStates.resetFor(top.getParent());
        }

        for (int i = 0; i < this.numRegionNodes; i++) {
            if (!this.regionNodes[i].isRoot()) {
                this.updateTransitionProbabilities(this.regionNodes[i]);
            }
        }

        this.numCladeNodes = 0;
        if (dataFreeClade != null) {
            this.computeCladePreOrder(dataFreeClade);
        }
    }

    /** Returns the number of branches of the region, by which a region history is indexed. */
    public int getNumBranches() {
        return this.numRegionNodes + this.numCladeNodes;
    }

    /**
     * Returns the node below the branch of the region with the given index: first the pruned
     * nodes in post-order with the top last, then the nodes of the data-free clade in pre-order.
     */
    public Node getBranchNode(int index) {
        return index < this.numRegionNodes ? this.regionNodes[index] : this.cladeNodes[index - this.numRegionNodes];
    }

    /** Creates an empty history of the region, with an empty list of mutations per branch. */
    public List<List<Mutation>> createEmptyHistory() {
        List<List<Mutation>> history = new ArrayList<>(this.getNumBranches());
        for (int i = 0; i < this.getNumBranches(); i++) {
            history.add(new ArrayList<>());
        }
        return history;
    }

    /**
     * Samples the history of the given site on the whole tree, indexed by the number of the
     * node below each branch. The mutations above the root encode the difference between the
     * reference and the sampled root state, and those of every other branch are sorted by
     * descending height. The region must be the whole tree, as set by updateModel.
     */
    public List<List<Mutation>> sampleSiteHistory(int site) {
        List<List<Mutation>> regionHistory = this.createEmptyHistory();
        this.sampleRegionSiteHistory(site, regionHistory);

        List<List<Mutation>> siteMutations = new ArrayList<>(this.tree.getNodeCount());
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            siteMutations.add(null);
        }
        for (int i = 0; i < this.numRegionNodes; i++) {
            siteMutations.set(this.regionNodes[i].getNr(), regionHistory.get(i));
        }
        return siteMutations;
    }

    /**
     * Samples the history of the given site on the region and adds its mutations to the given
     * lists, one per branch of the region, so that the histories of several sites can be
     * collected in the same lists. Above the root, the mutations encode the difference between
     * the reference and the sampled root state.
     */
    public void sampleRegionSiteHistory(int site, List<List<Mutation>> branchMutations) {
        Node top = this.regionNodes[this.numRegionNodes - 1];

        // sample the node states: prune upwards, then sample downwards starting at the top

        this.computePartials(site);

        double[] topWeights = this.stateWeights;
        double[] topPartial = this.partials[top.getNr()];
        if (top.isRoot()) {
            for (int state = 0; state < this.numStates; state++) {
                topWeights[state] = this.process.getFrequencies()[state] * topPartial[state];
            }
        } else {
            int parentState = this.topParentStates.applyAsInt(site);
            this.nodeStates[top.getParent().getNr()] = parentState;
            double[] topTransitionProbabilities = this.transitionProbabilities[top.getNr()];
            for (int state = 0; state < this.numStates; state++) {
                topWeights[state] = topTransitionProbabilities[parentState * this.numStates + state] * topPartial[state];
            }
        }
        this.nodeStates[top.getNr()] = StochasticMapping.sampleIndex(topWeights);

        this.sampleNodeStates();

        // sample the path on every pruned branch given its end states, then the data-free clade from the state above it

        this.sampleBranchHistories(site, branchMutations);
        if (this.dataFreeClade != null) {
            this.simulateClade(site, branchMutations);
        }
    }

    /** Returns the state of the given node in the history sampled last. */
    public int getNodeState(Node node) {
        return this.nodeStates[node.getNr()];
    }

    /**
     * Samples the paths of the given site on every branch of the region given the sampled node
     * states, like sampleBranchHistory on every branch, but without a random draw for most
     * branches. On a branch whose ends have the same state, the number of jumps of the
     * stochastic mapping is first drawn from the unconditioned Poisson(λ s t) distribution,
     * where λ is its jump rate, by placing a single Poisson process of rate 1 along the
     * expected numbers of jumps λ s t of all such branches laid end to end. A branch without
     * jumps then keeps its state without a mutation, and one with jumps is accepted with the
     * probability that the jumps return to the state, which makes n exact given the end
     * states, or else falls back to an exact draw. As the expected number of jumps on the
     * whole tree is small for most sites, only few random draws are needed.
     */
    private void sampleBranchHistories(int site, List<List<Mutation>> branchMutations) {
        double jumpRate = this.stochasticMapping.getJumpRate();

        // the position of the next jump on the axis of expected jumps, and how far the branches reach along it
        double nextJumpPosition = this.drawExponential();
        double branchesEndPosition = 0.0;

        for (int i = 0; i < this.numRegionNodes; i++) {
            Node node = this.regionNodes[i];
            int nodeNr = node.getNr();
            if (node.isRoot()) {
                branchMutations.get(i).addAll(this.createRootHistory(node, site));
                continue;
            }

            int state = this.nodeStates[nodeNr];
            if (state != this.nodeStates[node.getParent().getNr()]) {
                branchMutations.get(i).addAll(this.sampleBranchHistory(node, site));
                continue;
            }

            double startHeight = node.getParent().getHeight();
            double endHeight = node.getHeight();
            branchesEndPosition += jumpRate * this.rateScales[nodeNr] * Math.max(startHeight - endHeight, 0.0);

            int numJumps = 0;
            while (nextJumpPosition < branchesEndPosition) {
                numJumps++;
                nextJumpPosition += this.drawExponential();
            }

            if (numJumps == 0) {
                continue;
            }

            List<Mutation> sameStateMutations = this.stochasticMapping.sampleSameStateHistory(nodeNr, site, startHeight, endHeight, state, numJumps);
            branchMutations.get(i).addAll(sameStateMutations != null ? sameStateMutations : this.sampleBranchHistory(node, site));
        }
    }

    /** Draws from the exponential distribution with rate 1. */
    private double drawExponential() {
        return -Math.log(1.0 - Randomizer.nextDouble());
    }

    /**
     * Simulates the history of the given site through the data-free clade, starting from the
     * sampled state above it, exactly under the model, and adds its mutations to the lists of
     * the clade branches. The events of the uniformised process are spread over the branches
     * laid end to end, so that the site needs a single stream of events for the whole clade
     * instead of a draw per branch. As the branches come in pre-order, each branch is reached
     * after all branches above it, and each event is resolved in the state at its position:
     * the state after the last mutation on the branch, or else the end state of the closest
     * branch above with a mutation, or the start state of the clade.
     */
    private void simulateClade(int site, List<List<Mutation>> branchMutations) {
        this.simulationStamp++;
        int cladeStartState = this.nodeStates[this.dataFreeClade.getParent().getNr()];
        double eventRate = this.process.getMaxEscapeRate();
        double totalLength = this.cladeBranchStarts[this.numCladeNodes];

        int branch = -1;
        int state = cladeStartState;
        double previousHeight = Mutation.BRANCH_START;
        for (double position = Randomizer.nextExponential(eventRate); position < totalLength; position += Randomizer.nextExponential(eventRate)) {
            int index = this.findCladeBranch(position);
            Node node = this.cladeNodes[index];
            if (index != branch) {
                branch = index;
                state = this.getSimulatedStartState(node, cladeStartState);
                previousHeight = Mutation.BRANCH_START;
            }

            int newState = this.process.sampleUniformisedJump(state);
            if (newState == state) {
                continue;
            }

            double height = node.getParent().getHeight() - (position - this.cladeBranchStarts[index]) / this.rateScales[node.getNr()];
            branchMutations.get(this.numRegionNodes + index).add(new Mutation(node.getNr(), height, previousHeight, site, state, newState));

            previousHeight = height;
            state = newState;
            this.simulatedEndStates[node.getNr()] = newState;
            this.simulatedEndStamps[node.getNr()] = this.simulationStamp;
        }
    }

    /** Returns the index of the clade branch whose stretch of the branches laid end to end contains the given position. */
    private int findCladeBranch(double position) {
        int low = 0;
        int high = this.numCladeNodes - 1;
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            if (this.cladeBranchStarts[middle] <= position) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        return low;
    }

    /**
     * Returns the state at the top of the branch above the given clade node in the current
     * simulation: the end state of the closest ancestor within the clade whose branch got a
     * mutation, or the start state of the clade.
     */
    private int getSimulatedStartState(Node node, int cladeStartState) {
        for (Node ancestor = node; ancestor != this.dataFreeClade; ) {
            ancestor = ancestor.getParent();
            if (this.simulatedEndStamps[ancestor.getNr()] == this.simulationStamp) {
                return this.simulatedEndStates[ancestor.getNr()];
            }
        }
        return cladeStartState;
    }

    /**
     * Computes the log probability that sampleSiteHistory proposes the given history of the
     * given site on the whole tree, indexed like its result, up to the likelihood of the tip
     * data at the site, which is the same for every history. The region must be the whole tree.
     */
    public double computeLogSiteHistoryDensity(int site, List<List<Mutation>> siteMutations) {
        List<List<Mutation>> regionHistory = new ArrayList<>(this.getNumBranches());
        for (int i = 0; i < this.getNumBranches(); i++) {
            regionHistory.add(siteMutations.get(this.getBranchNode(i).getNr()));
        }
        return this.computeLogRegionSiteHistoryDensity(site, regionHistory);
    }

    /**
     * Collects the current history of the given site on the region, indexed like the
     * branches of the region, in the same form as a sampled history.
     */
    public List<List<Mutation>> collectRegionSiteHistory(int site) {
        List<List<Mutation>> regionHistory = new ArrayList<>(this.getNumBranches());
        for (int i = 0; i < this.getNumBranches(); i++) {
            List<Mutation> branchMutations = this.mutations.getMutations(this.getBranchNode(i));
            List<Mutation> siteMutations = new ArrayList<>();
            for (int j = 0; j < branchMutations.size(); j++) {
                if (branchMutations.get(j).site() == site) {
                    siteMutations.add(branchMutations.get(j));
                }
            }
            regionHistory.add(siteMutations);
        }
        return regionHistory;
    }

    /**
     * Computes the log probability that sampleRegionSiteHistory proposes the given history of
     * the given site on the region, indexed like its result, up to the likelihood of the data
     * and the fixed states, which is the same for every history. This is the frequency of the
     * root state if the root is the top, and the transition probability of every branch, which
     * give the probability of the node states, times the density of the stochastic mapping on
     * every branch. In the data-free clade, this is the density of the simulated paths if the
     * stochastic mapping is exact.
     */
    public double computeLogRegionSiteHistoryDensity(int site, List<List<Mutation>> regionHistory) {
        Node top = this.regionNodes[this.numRegionNodes - 1];

        // replay the history from the top downwards to find the state of every node

        double logDensity = 0.0;
        if (top.isRoot()) {
            int rootState = this.mutations.getReferenceSequence()[site];
            for (Mutation mutation : regionHistory.get(this.numRegionNodes - 1)) {
                rootState = mutation.newState();
            }
            this.nodeStates[top.getNr()] = rootState;
            logDensity += Math.log(this.process.getFrequencies()[rootState]);
        } else {
            this.nodeStates[top.getParent().getNr()] = this.topParentStates.applyAsInt(site);
        }

        // the pruned branches from the top down, then the clade branches, which come in pre-order

        for (int i = this.numRegionNodes - 1; i >= 0; i--) {
            if (!this.regionNodes[i].isRoot()) {
                logDensity += this.computeLogBranchDensity(this.regionNodes[i], regionHistory.get(i));
            }
        }
        for (int i = 0; i < this.numCladeNodes; i++) {
            logDensity += this.computeLogBranchDensity(this.cladeNodes[i], regionHistory.get(this.numRegionNodes + i));
        }

        return logDensity;
    }

    /**
     * Returns the log of the transition probability of the given branch times the density of
     * its given path, starting from the state of its parent, and sets the state of its node.
     */
    private double computeLogBranchDensity(Node node, List<Mutation> branchMutations) {
        int nodeNr = node.getNr();
        Node parent = node.getParent();

        int startState = this.nodeStates[parent.getNr()];
        int endState = branchMutations.isEmpty() ? startState : branchMutations.get(branchMutations.size() - 1).newState();
        this.nodeStates[nodeNr] = endState;

        double endProbability = this.transitionProbabilities[nodeNr][startState * this.numStates + endState];

        return Math.log(endProbability) + this.stochasticMapping.computeLogSiteHistoryDensity(
                branchMutations, parent.getHeight(), node.getHeight(),
                startState, endState, this.rateScales[nodeNr], endProbability
        );
    }

    /* Node States */

    /**
     * Computes the partial likelihoods of every node of the region at the given site. A tip
     * allows every state compatible with its observed character, and a boundary node only its
     * current state. The partials of the inner nodes are rescaled, which does not change the
     * sampled states.
     */
    private void computePartials(int site) {
        int patternIndex = this.alignment.getPatternIndex(site);
        if (this.dataFreeClade != null) {
            this.knownStates[this.dataFreeClade.getNr()] = UNINFORMATIVE;
        }

        int boundaryIndex = 0;
        for (int i = 0; i < this.numRegionNodes; i++) {
            Node node = this.regionNodes[i];
            if (this.boundaryStamps[node.getNr()] == this.regionStamp) {
                this.setKnownState(node, this.boundaryStates.get(boundaryIndex++).applyAsInt(site));
            } else {
                this.computePartial(node, patternIndex);
            }
        }
    }

    /**
     * Lets the given node allow only the given state. Its partial likelihoods are not set, as
     * the partials of a node with a known state are never read.
     */
    private void setKnownState(Node node, int state) {
        this.knownStates[node.getNr()] = state;
    }

    /**
     * Computes the partial likelihoods of the given node at a site with the given pattern
     * from the partial likelihoods of its children. A child that allows every state equally
     * contributes a factor of one, as the rows of the transition probabilities sum to one. The
     * partials of an inner node are rescaled by their maximum to avoid underflow on large trees.
     */
    private void computePartial(Node node, int patternIndex) {
        double[] partial = this.partials[node.getNr()];

        if (node.isLeaf()) {
            int code = this.alignment.getPattern(this.taxonIndices[node.getNr()], patternIndex);
            System.arraycopy(this.getTipPartial(code), 0, partial, 0, this.numStates);
            this.knownStates[node.getNr()] = this.tipStates[code];
            return;
        }

        this.knownStates[node.getNr()] = UNINFORMATIVE;
        Arrays.fill(partial, 1.0);

        // index the children directly, as getChildren wraps them in a new view on every call

        for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
            Node child = node.getChild(childIndex);
            int knownChildState = this.knownStates[child.getNr()];
            if (knownChildState == UNINFORMATIVE) {
                continue;
            }

            this.knownStates[node.getNr()] = -1;
            double[] childPartial = this.partials[child.getNr()];
            double[] childTransitionProbabilities = this.transitionProbabilities[child.getNr()];

            // a child that allows a single state, like most tips, only needs the column of that state

            if (knownChildState >= 0) {
                for (int parentState = 0; parentState < this.numStates; parentState++) {
                    partial[parentState] *= childTransitionProbabilities[parentState * this.numStates + knownChildState];
                }
                continue;
            }

            for (int parentState = 0; parentState < this.numStates; parentState++) {
                double sum = 0.0;
                for (int childState = 0; childState < this.numStates; childState++) {
                    sum += childTransitionProbabilities[parentState * this.numStates + childState] * childPartial[childState];
                }
                partial[parentState] *= sum;
            }
        }

        if (this.knownStates[node.getNr()] == UNINFORMATIVE) {
            return;
        }

        // rescale to avoid underflow on large trees

        double maxPartial = 0.0;
        for (double value : partial) {
            maxPartial = Math.max(maxPartial, value);
        }
        for (int state = 0; state < this.numStates; state++) {
            partial[state] /= maxPartial;
        }
    }

    /**
     * Samples the states of every node of the region below the top, given the state of the
     * top, from the top down. A child state is drawn proportional to the transition
     * probability from the parent state times the partial likelihood of the child, unless the
     * partial likelihood only allows a single state, like at most tips and at the boundary.
     */
    private void sampleNodeStates() {
        double[] weights = this.stateWeights;

        for (int i = this.numRegionNodes - 2; i >= 0; i--) {
            Node node = this.regionNodes[i];

            int knownState = this.knownStates[node.getNr()];
            if (knownState >= 0) {
                this.nodeStates[node.getNr()] = knownState;
                continue;
            }

            int parentState = this.nodeStates[node.getParent().getNr()];
            double[] partial = this.partials[node.getNr()];
            double[] branchTransitionProbabilities = this.transitionProbabilities[node.getNr()];

            for (int state = 0; state < this.numStates; state++) {
                weights[state] = branchTransitionProbabilities[parentState * this.numStates + state] * partial[state];
            }

            this.nodeStates[node.getNr()] = StochasticMapping.sampleIndex(weights);
        }
    }

    /**
     * Updates the scale of Q on the branch above the given node and recomputes the
     * transition probabilities P = exp(Q t) of the branch if its scaled length t has changed.
     */
    private void updateTransitionProbabilities(Node node) {
        int nodeNr = node.getNr();

        this.rateScales[nodeNr] = this.process.getRateScale(node);
        double duration = this.rateScales[nodeNr] * (node.getParent().getHeight() - node.getHeight());

        if (duration != this.durations[nodeNr]) {
            this.process.computeTransitionProbabilities(duration, this.transitionProbabilities[nodeNr]);
            this.durations[nodeNr] = duration;
        }
    }

    /* Branch Histories */

    /**
     * Samples the history of the given site on the branch above the given node, given the
     * sampled states at both ends. The rate matrix of the branch is Q scaled by the branch
     * rate and the site rate.
     */
    private List<Mutation> sampleBranchHistory(Node node, int site) {
        int nodeNr = node.getNr();
        Node parent = node.getParent();
        int startState = this.nodeStates[parent.getNr()];
        int endState = this.nodeStates[nodeNr];

        double endProbability = this.transitionProbabilities[nodeNr][startState * this.numStates + endState];

        return this.stochasticMapping.sampleSiteHistory(
                nodeNr, site, parent.getHeight(), node.getHeight(),
                startState, endState, this.rateScales[nodeNr], endProbability
        );
    }

    /**
     * Creates the history of the given site above the root, which encodes the difference
     * between the reference and the sampled root state as a single mutation at the root.
     */
    private List<Mutation> createRootHistory(Node root, int site) {
        int referenceState = this.mutations.getReferenceSequence()[site];
        int rootState = this.nodeStates[root.getNr()];

        List<Mutation> rootMutations = new ArrayList<>();
        if (rootState != referenceState) {
            rootMutations.add(new Mutation(root.getNr(), root.getHeight(), root.getHeight(), site, referenceState, rootState));
        }
        return rootMutations;
    }

    /* Helpers */

    /**
     * Stores the nodes of the region below the given top in post-order, iteratively so that
     * deep trees cannot overflow the stack, and stamps the boundary nodes, the children of the
     * top and of free nodes that are not free themselves.
     */
    private void computeRegionPostOrder(Node top, Predicate<Node> isFree) {
        // collect the nodes in pre-order with the children in reverse, then reverse the whole order

        int numVisited = 0;
        Node[] stack = this.traversalStack;
        int stackSize = 0;
        stack[stackSize++] = top;

        while (stackSize > 0) {
            Node node = stack[--stackSize];
            this.regionNodes[numVisited++] = node;
            if (this.boundaryStamps[node.getNr()] == this.regionStamp) {
                continue;
            }

            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                Node child = node.getChild(childIndex);
                if (child == this.dataFreeClade) {
                    continue;
                }
                if (!isFree.test(child)) {
                    this.boundaryStamps[child.getNr()] = this.regionStamp;
                }
                stack[stackSize++] = child;
            }
        }

        this.numRegionNodes = numVisited;
        for (int i = 0; i < numVisited / 2; i++) {
            Node node = this.regionNodes[i];
            this.regionNodes[i] = this.regionNodes[numVisited - 1 - i];
            this.regionNodes[numVisited - 1 - i] = node;
        }

        // the boundary nodes in the post-order, so that computePartials meets them in the order of their lookups

        for (int i = 0; i < numVisited; i++) {
            if (this.boundaryStamps[this.regionNodes[i].getNr()] == this.regionStamp) {
                this.boundaryNodes.add(this.regionNodes[i]);
            }
        }
    }

    /**
     * Stores the nodes of the given data-free clade in pre-order, with every node after its
     * parent, together with the starts of their branches laid end to end in scaled time, and
     * computes the transition probabilities of the branches.
     */
    private void computeCladePreOrder(Node clade) {
        this.cladeNodes[0] = clade;
        this.numCladeNodes = 1;

        // the array grows while it is traversed, which yields the breadth-first order
        for (int i = 0; i < this.numCladeNodes; i++) {
            Node node = this.cladeNodes[i];
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                this.cladeNodes[this.numCladeNodes++] = node.getChild(childIndex);
            }

            this.updateTransitionProbabilities(node);
            this.cladeBranchStarts[i + 1] = this.cladeBranchStarts[i] + Math.max(this.durations[node.getNr()], 0.0);
        }
    }

    /**
     * Returns the tip partial likelihoods of the given character code: 1 for every compatible
     * state and 0 otherwise. Also caches the only compatible state of the code in tipStates,
     * -1 if there are several, or UNINFORMATIVE if all states are compatible.
     */
    private double[] getTipPartial(int code) {
        if (code >= this.tipPartials.length) {
            this.tipPartials = Arrays.copyOf(this.tipPartials, code + 1);
            this.tipStates = Arrays.copyOf(this.tipStates, code + 1);
        }

        if (this.tipPartials[code] == null) {
            int[] compatibleStates = this.dataType.getStatesForCode(code);

            double[] partial = new double[this.numStates];
            for (int state : compatibleStates) {
                partial[state] = 1.0;
            }
            this.tipPartials[code] = partial;
            this.tipStates[code] = compatibleStates.length == 1 ? compatibleStates[0]
                    : compatibleStates.length == this.numStates ? UNINFORMATIVE : -1;
        }

        return this.tipPartials[code];
    }

}
