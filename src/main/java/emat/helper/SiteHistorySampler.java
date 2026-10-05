package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.UniformisedStochasticMapping;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Samples the complete history of a single site on the tree from its exact conditional
 * distribution under the model of the genetic prior, given the tip data (stochastic
 * mapping). Per site, the genetic prior is the density of a CTMC path on the tree, so a draw
 * consists of sampling the node states by pruning and then the path on every branch given
 * its end states. The probability of a history is its genetic prior divided by the
 * likelihood of the tip data, which computeLogLikelihood provides. The node states are
 * always sampled exactly, while the paths on the branches are sampled by the given
 * stochastic mapping, which may approximate the model.
 */
public class SiteHistorySampler {

    // the product of rescaling factors below which computePartials takes its log, far above underflow
    private static final double MIN_SCALE = 1e-200;

    final Mutations mutations;
    final GeneticPrior geneticPrior;
    final TreeInterface tree;
    final Alignment alignment;
    final DataType dataType;
    final StochasticMapping stochasticMapping;

    final int numStates;

    // the model of the current tree: Q, its eigen decomposition, the site rate, and the root frequencies
    double[] rateMatrix;
    EigenDecomposition eigenDecomposition;
    double siteRate;
    double[] frequencies;

    // the nodes of the current tree in post-order, so that every child precedes its parent,
    // and the stack used to collect them
    final Node[] postOrderNodes;
    final Node[] traversalStack;

    // per tip node: the index of its taxon in the alignment
    final int[] taxonIndices;

    // per character code: the tip partial likelihoods, created when first needed
    double[][] tipPartials = new double[0][];

    // per node: the scale of Q on the branch above it, the scaled branch length for which
    // the transition probabilities were computed, the transition probabilities of the
    // branch, the partial likelihoods of the subtree below it, and its sampled state
    final double[] rateScales;
    final double[] durations;
    final double[][] transitionProbabilities;
    final double[][] partials;
    final int[] nodeStates;

    public SiteHistorySampler(Mutations mutations, GeneticPrior geneticPrior, StochasticMapping stochasticMapping) {
        this.mutations = mutations;
        this.geneticPrior = geneticPrior;
        this.tree = mutations.getTree();
        this.alignment = mutations.getAlignment();
        this.dataType = this.alignment.getDataType();

        this.numStates = this.alignment.getMaxStateCount();
        this.stochasticMapping = stochasticMapping;

        int numNodes = this.tree.getNodeCount();
        this.postOrderNodes = new Node[numNodes];
        this.traversalStack = new Node[numNodes];
        this.rateScales = new double[numNodes];
        this.durations = new double[numNodes];
        this.transitionProbabilities = new double[numNodes][this.numStates * this.numStates];
        this.partials = new double[numNodes][this.numStates];
        this.nodeStates = new int[numNodes];

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
     * Takes the current model from the genetic prior and computes the transition
     * probabilities of every branch of the current tree. This has to be called again
     * whenever the tree or the model changes. The transition probabilities of a branch are
     * only recomputed if Q or the scaled length of the branch has changed.
     */
    public void updateModel() {
        Node root = this.tree.getRoot();

        this.updateRateMatrix();
        this.computePostOrder(root);

        for (Node node : this.postOrderNodes) {
            if (!node.isRoot()) {
                this.updateTransitionProbabilities(node);
            }
        }
    }

    /**
     * Takes the current model from the genetic prior and computes the transition
     * probabilities of the branches above the given nodes only, skipping the root. This is
     * a cheaper alternative to updateModel for sampling the history on a part of the tree.
     */
    public void updateModel(List<Node> branchNodes) {
        this.updateRateMatrix();

        for (Node node : branchNodes) {
            if (!node.isRoot()) {
                this.updateTransitionProbabilities(node);
            }
        }
    }

    /**
     * Computes the log likelihood of the tip data summed over all sites, i.e. the genetic
     * prior marginalised over all histories that are compatible with the tips.
     */
    public double computeLogLikelihood() {
        double logLikelihood = 0.0;
        for (int patternIndex = 0; patternIndex < this.alignment.getPatternCount(); patternIndex++) {
            logLikelihood += this.alignment.getPatternWeight(patternIndex) * this.computePatternLogLikelihood(patternIndex);
        }
        return logLikelihood;
    }

    /**
     * Samples the history of the given site, indexed by the number of the node below each
     * branch. The mutations above the root encode the difference between the reference and
     * the sampled root state, and those of every other branch are sorted by descending
     * height.
     */
    public List<List<Mutation>> sampleSiteHistory(int site) {
        Node root = this.tree.getRoot();

        // sample the node states: prune upwards, then sample downwards starting at the root

        this.computePartials(this.alignment.getPatternIndex(site));

        double[] rootWeights = new double[this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            rootWeights[state] = this.frequencies[state] * this.partials[root.getNr()][state];
        }
        this.nodeStates[root.getNr()] = StochasticMapping.sampleIndex(rootWeights);

        this.sampleNodeStates();

        // sample the path on every branch given its end states

        if (this.stochasticMapping instanceof UniformisedStochasticMapping uniformisedMapping) {
            return this.sampleUniformisedBranchHistories(site, uniformisedMapping);
        }

        List<List<Mutation>> siteMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            siteMutations.add(node.isRoot() ? this.createRootHistory(node, site) : this.sampleBranchHistory(node, site));
        }

        return siteMutations;
    }

    /**
     * Samples the paths of the given site on every branch given the sampled node states,
     * like sampleBranchHistory on every branch, but without a random draw for most branches.
     * On a branch whose ends have the same state, the number of uniformised jumps is first
     * drawn from the unconditioned Poisson(μ* s t) distribution, by placing a single Poisson
     * process of rate 1 along the expected numbers of jumps μ* s t of all such branches laid
     * end to end. A branch without jumps then keeps its state without a mutation, and one
     * with jumps is accepted with probability (B^n)_{ii}, which makes n exact given the end
     * states, or else falls back to an exact draw. As the expected number of jumps on the
     * whole tree is small for most sites, only few random draws are needed.
     */
    private List<List<Mutation>> sampleUniformisedBranchHistories(int site, UniformisedStochasticMapping uniformisedMapping) {
        double uniformisationRate = uniformisedMapping.getUniformisationRate();

        // the position of the next jump on the axis of expected jumps, and how far the branches reach along it
        double nextJumpPosition = this.drawExponential();
        double branchesEndPosition = 0.0;

        List<List<Mutation>> siteMutations = new ArrayList<>(this.tree.getNodeCount());
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            if (node.isRoot()) {
                siteMutations.add(this.createRootHistory(node, site));
                continue;
            }

            int state = this.nodeStates[nodeNr];
            if (state != this.nodeStates[node.getParent().getNr()]) {
                siteMutations.add(this.sampleBranchHistory(node, site));
                continue;
            }

            double startHeight = node.getParent().getHeight();
            double endHeight = node.getHeight();
            branchesEndPosition += uniformisationRate * this.rateScales[nodeNr] * Math.max(startHeight - endHeight, 0.0);

            int numJumps = 0;
            while (nextJumpPosition < branchesEndPosition) {
                numJumps++;
                nextJumpPosition += this.drawExponential();
            }

            if (numJumps == 0) {
                siteMutations.add(List.of());
                continue;
            }

            List<Mutation> branchMutations = uniformisedMapping.sampleSameStateHistory(nodeNr, site, startHeight, endHeight, state, numJumps);
            siteMutations.add(branchMutations != null ? branchMutations : this.sampleBranchHistory(node, site));
        }

        return siteMutations;
    }

    /** Draws from the exponential distribution with rate 1. */
    private double drawExponential() {
        return -Math.log(1.0 - Randomizer.nextDouble());
    }

    /**
     * Computes the log probability that sampleSiteHistory proposes the given history of the
     * given site, indexed like its result, up to the likelihood of the tip data at the site,
     * which is the same for every history. This is the root frequency of the root state and
     * the transition probability of every branch, which give the probability of the node
     * states, times the density of the stochastic mapping on every branch.
     */
    public double computeLogSiteHistoryDensity(int site, List<List<Mutation>> siteMutations) {
        Node root = this.tree.getRoot();

        // replay the history from the reference downwards to find the state of every node

        int rootState = this.mutations.getReferenceSequence()[site];
        for (Mutation mutation : siteMutations.get(root.getNr())) {
            rootState = mutation.newState();
        }
        this.nodeStates[root.getNr()] = rootState;

        double logDensity = Math.log(this.frequencies[rootState]);

        for (int i = this.postOrderNodes.length - 1; i >= 0; i--) {
            Node node = this.postOrderNodes[i];
            if (node.isRoot()) {
                continue;
            }

            int nodeNr = node.getNr();
            Node parent = node.getParent();
            List<Mutation> branchMutations = siteMutations.get(nodeNr);

            int startState = this.nodeStates[parent.getNr()];
            int endState = branchMutations.isEmpty() ? startState : branchMutations.get(branchMutations.size() - 1).newState();
            this.nodeStates[nodeNr] = endState;

            double endProbability = this.transitionProbabilities[nodeNr][startState * this.numStates + endState];

            logDensity += Math.log(endProbability);
            logDensity += this.stochasticMapping.computeLogSiteHistoryDensity(
                    branchMutations, parent.getHeight(), node.getHeight(),
                    startState, endState, this.rateScales[nodeNr], endProbability
            );
        }

        return logDensity;
    }

    /**
     * Samples the history of the given site on a connected region of branches from its exact
     * conditional distribution given the histories on all other branches. The region is
     * given by the nodes below its branches, starting with its top node and listing every
     * other node after its parent. The given start state at the top of the branch above the
     * top node stays fixed, and is ignored if the top node is the root. A node with a
     * non-negative fixed state keeps this state, which cuts the region off from the branches
     * below it. Every other inner node must have all of its children in the region, and
     * every other tip keeps a state compatible with its data. The node states are sampled by
     * pruning within the region, then the path on every branch given its end states. The
     * returned histories are indexed like the region nodes. updateModel must have been
     * called for the branches of the region.
     */
    public List<List<Mutation>> sampleRegionHistory(int site, int startState, List<Node> regionNodes, int[] fixedStates) {
        Node top = regionNodes.getFirst();
        this.computeRegionPartials(site, regionNodes, fixedStates);

        // sample the node states from the top down

        double[] weights = new double[this.numStates];
        for (int i = 0; i < regionNodes.size(); i++) {
            Node node = regionNodes.get(i);

            if (fixedStates[i] >= 0) {
                this.nodeStates[node.getNr()] = fixedStates[i];
            } else {
                this.computeNodeStateWeights(node, node == top ? startState : this.nodeStates[node.getParent().getNr()], weights);
                this.nodeStates[node.getNr()] = StochasticMapping.sampleIndex(weights);
            }
        }

        // sample the path on every branch of the region given its end states

        if (!top.isRoot()) {
            this.nodeStates[top.getParent().getNr()] = startState;
        }

        List<List<Mutation>> regionMutations = new ArrayList<>();
        for (Node node : regionNodes) {
            regionMutations.add(node.isRoot() ? this.createRootHistory(node, site) : this.sampleBranchHistory(node, site));
        }

        return regionMutations;
    }

    /**
     * Computes the log probability that sampleRegionHistory proposes the given history of
     * the given site on the region, indexed like the region nodes, given the same start
     * state and fixed states. This is the probability of the states of the nodes that are
     * not fixed times the density of the stochastic mapping on every branch of the region.
     * The history above the root follows from the root state, so it adds nothing.
     */
    public double computeLogRegionHistoryDensity(int site, int startState, List<Node> regionNodes, int[] fixedStates,
                                                 List<List<Mutation>> regionMutations) {
        Node top = regionNodes.getFirst();
        this.computeRegionPartials(site, regionNodes, fixedStates);

        double logDensity = 0.0;
        double[] weights = new double[this.numStates];

        for (int i = 0; i < regionNodes.size(); i++) {
            Node node = regionNodes.get(i);
            List<Mutation> branchMutations = regionMutations.get(i);

            // the state of a node is set by the last mutation above it, or else equals the state above its branch

            int branchStartState;
            if (node.isRoot()) {
                branchStartState = this.mutations.getReferenceSequence()[site];
            } else {
                branchStartState = node == top ? startState : this.nodeStates[node.getParent().getNr()];
            }
            int state = branchMutations.isEmpty() ? branchStartState : branchMutations.getLast().newState();
            this.nodeStates[node.getNr()] = state;

            if (fixedStates[i] < 0) {
                this.computeNodeStateWeights(node, branchStartState, weights);
                double totalWeight = 0.0;
                for (double weight : weights) {
                    totalWeight += weight;
                }
                logDensity += Math.log(weights[state] / totalWeight);
            }

            if (!node.isRoot()) {
                logDensity += this.computeLogBranchHistoryDensity(node, branchMutations, branchStartState, state);
            }
        }

        return logDensity;
    }

    /* Node States */

    /**
     * Computes the partial likelihoods of the given site for every node of a region, whose
     * nodes are given with every node after its parent. A node with a fixed state only
     * allows this state, and a tip every state compatible with its data.
     */
    private void computeRegionPartials(int site, List<Node> regionNodes, int[] fixedStates) {
        int patternIndex = this.alignment.getPatternIndex(site);

        for (int i = regionNodes.size() - 1; i >= 0; i--) {
            Node node = regionNodes.get(i);

            if (fixedStates[i] >= 0) {
                double[] partial = this.partials[node.getNr()];
                Arrays.fill(partial, 0.0);
                partial[fixedStates[i]] = 1.0;
            } else {
                this.computePartial(node, patternIndex);
            }
        }
    }

    /**
     * Computes the unnormalised probabilities of the states of the given node given the
     * state at the start of the branch above it and the tip data below it: the transition
     * probability from the start state, or the root frequency if the node is the root, times
     * the partial likelihood of the node.
     */
    private void computeNodeStateWeights(Node node, int startState, double[] weights) {
        double[] partial = this.partials[node.getNr()];
        for (int state = 0; state < this.numStates; state++) {
            double priorProbability = node.isRoot()
                    ? this.frequencies[state]
                    : this.transitionProbabilities[node.getNr()][startState * this.numStates + state];
            weights[state] = priorProbability * partial[state];
        }
    }

    /** Computes the log likelihood of the tip data at a single site with the given pattern. */
    private double computePatternLogLikelihood(int patternIndex) {
        Node root = this.tree.getRoot();
        double logScale = this.computePartials(patternIndex);

        double likelihood = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            likelihood += this.frequencies[state] * this.partials[root.getNr()][state];
        }

        return logScale + Math.log(likelihood);
    }

    /**
     * Computes the partial likelihoods of every node at a site with the given pattern. A
     * tip allows every state compatible with its observed character. The partials of the
     * inner nodes are rescaled, which does not change the sampled states, and the log of
     * all rescaling factors is returned.
     */
    private double computePartials(int patternIndex) {
        // multiply the rescaling factors and only take the log when their product gets small

        double logScale = 0.0;
        double scale = 1.0;
        for (Node node : this.postOrderNodes) {
            scale *= this.computePartial(node, patternIndex);
            if (scale < MIN_SCALE) {
                logScale += Math.log(scale);
                scale = 1.0;
            }
        }
        return logScale + Math.log(scale);
    }

    /**
     * Computes the partial likelihoods of the given node at a site with the given pattern
     * from the partial likelihoods of its children. The partials of an inner node are
     * rescaled by their maximum, which is at most 1 and is returned. A tip returns 1.
     */
    private double computePartial(Node node, int patternIndex) {
        double[] partial = this.partials[node.getNr()];

        if (node.isLeaf()) {
            int code = this.alignment.getPattern(this.taxonIndices[node.getNr()], patternIndex);
            System.arraycopy(this.getTipPartial(code), 0, partial, 0, this.numStates);
            return 1.0;
        }

        Arrays.fill(partial, 1.0);

        // index the children directly, as getChildren wraps them in a new view on every call

        for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
            Node child = node.getChild(childIndex);
            double[] childPartial = this.partials[child.getNr()];
            double[] childTransitionProbabilities = this.transitionProbabilities[child.getNr()];

            for (int parentState = 0; parentState < this.numStates; parentState++) {
                double sum = 0.0;
                for (int childState = 0; childState < this.numStates; childState++) {
                    sum += childTransitionProbabilities[parentState * this.numStates + childState] * childPartial[childState];
                }
                partial[parentState] *= sum;
            }
        }

        // rescale to avoid underflow on large trees

        double maxPartial = 0.0;
        for (double value : partial) {
            maxPartial = Math.max(maxPartial, value);
        }
        for (int state = 0; state < this.numStates; state++) {
            partial[state] /= maxPartial;
        }

        return maxPartial;
    }

    /**
     * Samples the states of every node below the root, given the state of the root, from
     * the top down. A child state is drawn proportional to the transition probability from
     * the parent state times the partial likelihood of the child.
     */
    private void sampleNodeStates() {
        double[] weights = new double[this.numStates];

        for (int i = this.postOrderNodes.length - 1; i >= 0; i--) {
            Node node = this.postOrderNodes[i];
            if (node.isRoot()) {
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

    /** Takes Q, the site rate and the root frequencies from the genetic prior. */
    private void updateRateMatrix() {
        Node root = this.tree.getRoot();

        // the substitution model may overwrite its eigen decomposition objects in place, so
        // they are fetched again every time and only their values are cached through Q

        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);

        double[] newRateMatrix = this.geneticPrior.computeRateMatrix();
        if (!Arrays.equals(newRateMatrix, this.rateMatrix)) {
            this.rateMatrix = newRateMatrix;
            this.stochasticMapping.setRateMatrix(this.rateMatrix);
            Arrays.fill(this.durations, Double.NaN);
        }
        this.siteRate = this.geneticPrior.siteModel.getRateForCategory(0, root);
        this.frequencies = this.geneticPrior.substitutionModel.getFrequencies();
    }

    /**
     * Updates the scale of Q on the branch above the given node and recomputes the
     * transition probabilities P = exp(Q t) of the branch if its scaled length t has changed.
     */
    private void updateTransitionProbabilities(Node node) {
        int nodeNr = node.getNr();

        this.rateScales[nodeNr] = this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
        double duration = this.rateScales[nodeNr] * (node.getParent().getHeight() - node.getHeight());

        if (duration != this.durations[nodeNr]) {
            StochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, duration, this.transitionProbabilities[nodeNr]);
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
     * Computes the log density of the stochastic mapping of the given history of a single
     * site on the branch above the given node, given the states at both ends.
     */
    private double computeLogBranchHistoryDensity(Node node, List<Mutation> branchMutations, int startState, int endState) {
        int nodeNr = node.getNr();
        return this.stochasticMapping.computeLogSiteHistoryDensity(
                branchMutations, node.getParent().getHeight(), node.getHeight(),
                startState, endState, this.rateScales[nodeNr],
                this.transitionProbabilities[nodeNr][startState * this.numStates + endState]
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
     * Stores the nodes of the tree below the given root in post-order, iteratively so that
     * deep trees cannot overflow the stack.
     */
    private void computePostOrder(Node root) {
        // collect the nodes in pre-order with the children in reverse, then reverse the whole order

        int numVisited = 0;
        Node[] stack = this.traversalStack;
        int stackSize = 0;
        stack[stackSize++] = root;

        while (stackSize > 0) {
            Node node = stack[--stackSize];
            this.postOrderNodes[this.postOrderNodes.length - 1 - numVisited++] = node;
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                stack[stackSize++] = node.getChild(childIndex);
            }
        }
    }

    /** Returns the tip partial likelihoods of the given character code: 1 for every compatible state and 0 otherwise. */
    private double[] getTipPartial(int code) {
        if (code >= this.tipPartials.length) {
            this.tipPartials = Arrays.copyOf(this.tipPartials, code + 1);
        }

        if (this.tipPartials[code] == null) {
            double[] partial = new double[this.numStates];
            for (int state : this.dataType.getStatesForCode(code)) {
                partial[state] = 1.0;
            }
            this.tipPartials[code] = partial;
        }

        return this.tipPartials[code];
    }

}
