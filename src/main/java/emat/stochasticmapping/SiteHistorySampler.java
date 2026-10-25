package emat.stochasticmapping;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;

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
 * its end states. The node states are always sampled exactly, while the paths on the
 * branches are sampled by the given stochastic mapping, which may approximate the model.
 */
public class SiteHistorySampler {

    final Mutations mutations;
    final SubstitutionProcess process;
    final TreeInterface tree;
    final Alignment alignment;
    final DataType dataType;
    final StochasticMapping stochasticMapping;

    final int numStates;

    // the rate matrix Q that the stochastic mapping and the cached transition probabilities were computed for
    double[] rateMatrix;

    // the nodes of the current tree in post-order, so that every child precedes its parent,
    // and the stack used to collect them
    final Node[] postOrderNodes;
    final Node[] traversalStack;

    // per tip node: the index of its taxon in the alignment
    final int[] taxonIndices;

    // per character code: the tip partial likelihoods, and the only state compatible with
    // it or -1 if there are several, created when first needed
    double[][] tipPartials = new double[0][];
    int[] tipStates = new int[0];

    // per node: the scale of Q on the branch above it, the scaled branch length for which
    // the transition probabilities were computed, the transition probabilities of the
    // branch, the partial likelihoods of the subtree below it, the only state its partial
    // likelihoods allow or -1 if they allow several, and its sampled state
    final double[] rateScales;
    final double[] durations;
    final double[][] transitionProbabilities;
    final double[][] partials;
    final int[] knownStates;
    final int[] nodeStates;

    public SiteHistorySampler(Mutations mutations, SubstitutionProcess process, StochasticMapping stochasticMapping) {
        this.mutations = mutations;
        this.process = process;
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
        this.knownStates = new int[numNodes];
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
     * Takes the current model from the substitution process and computes the transition
     * probabilities of every branch of the current tree. This has to be called again
     * whenever the tree or the model changes. The transition probabilities of a branch are
     * only recomputed if Q or the scaled length of the branch has changed.
     */
    public void updateModel() {
        Node root = this.tree.getRoot();

        this.process.update();
        if (this.process.getRateMatrix() != this.rateMatrix) {
            this.rateMatrix = this.process.getRateMatrix();
            this.stochasticMapping.setRateMatrix(this.rateMatrix);
            Arrays.fill(this.durations, Double.NaN);
        }
        this.computePostOrder(root);

        for (Node node : this.postOrderNodes) {
            if (!node.isRoot()) {
                this.updateTransitionProbabilities(node);
            }
        }
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
            rootWeights[state] = this.process.getFrequencies()[state] * this.partials[root.getNr()][state];
        }
        this.nodeStates[root.getNr()] = StochasticMapping.sampleIndex(rootWeights);

        this.sampleNodeStates();

        // sample the path on every branch given its end states

        return this.sampleBranchHistories(site);
    }

    /**
     * Samples the paths of the given site on every branch given the sampled node states,
     * like sampleBranchHistory on every branch, but without a random draw for most branches.
     * On a branch whose ends have the same state, the number of jumps of the stochastic
     * mapping is first drawn from the unconditioned Poisson(λ s t) distribution, where λ is
     * its jump rate, by placing a single Poisson process of rate 1 along the expected numbers
     * of jumps λ s t of all such branches laid end to end. A branch without jumps then keeps
     * its state without a mutation, and one with jumps is accepted with the probability that
     * the jumps return to the state, which makes n exact given the end states, or else falls
     * back to an exact draw. As the expected number of jumps on the whole tree is small for
     * most sites, only few random draws are needed.
     */
    private List<List<Mutation>> sampleBranchHistories(int site) {
        double jumpRate = this.stochasticMapping.getJumpRate();

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
            branchesEndPosition += jumpRate * this.rateScales[nodeNr] * Math.max(startHeight - endHeight, 0.0);

            int numJumps = 0;
            while (nextJumpPosition < branchesEndPosition) {
                numJumps++;
                nextJumpPosition += this.drawExponential();
            }

            if (numJumps == 0) {
                siteMutations.add(List.of());
                continue;
            }

            List<Mutation> branchMutations = this.stochasticMapping.sampleSameStateHistory(nodeNr, site, startHeight, endHeight, state, numJumps);
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

        double logDensity = Math.log(this.process.getFrequencies()[rootState]);

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

    /* Node States */

    /**
     * Computes the partial likelihoods of every node at a site with the given pattern. A
     * tip allows every state compatible with its observed character. The partials of the
     * inner nodes are rescaled, which does not change the sampled states.
     */
    private void computePartials(int patternIndex) {
        for (Node node : this.postOrderNodes) {
            this.computePartial(node, patternIndex);
        }
    }

    /**
     * Computes the partial likelihoods of the given node at a site with the given pattern
     * from the partial likelihoods of its children. The partials of an inner node are
     * rescaled by their maximum to avoid underflow on large trees.
     */
    private void computePartial(Node node, int patternIndex) {
        double[] partial = this.partials[node.getNr()];

        if (node.isLeaf()) {
            int code = this.alignment.getPattern(this.taxonIndices[node.getNr()], patternIndex);
            System.arraycopy(this.getTipPartial(code), 0, partial, 0, this.numStates);
            this.knownStates[node.getNr()] = this.tipStates[code];
            return;
        }

        this.knownStates[node.getNr()] = -1;
        Arrays.fill(partial, 1.0);

        // index the children directly, as getChildren wraps them in a new view on every call

        for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
            Node child = node.getChild(childIndex);
            double[] childPartial = this.partials[child.getNr()];
            double[] childTransitionProbabilities = this.transitionProbabilities[child.getNr()];

            // a child that allows a single state, like most tips, only needs the column of that state

            int knownChildState = this.knownStates[child.getNr()];
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
     * Samples the states of every node below the root, given the state of the root, from
     * the top down. A child state is drawn proportional to the transition probability from
     * the parent state times the partial likelihood of the child, unless the partial
     * likelihood only allows a single state, like at most tips.
     */
    private void sampleNodeStates() {
        double[] weights = new double[this.numStates];

        for (int i = this.postOrderNodes.length - 1; i >= 0; i--) {
            Node node = this.postOrderNodes[i];
            if (node.isRoot()) {
                continue;
            }

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

    /**
     * Returns the tip partial likelihoods of the given character code: 1 for every compatible
     * state and 0 otherwise. Also caches the only compatible state of the code in tipStates,
     * or -1 if there are several.
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
            this.tipStates[code] = compatibleStates.length == 1 ? compatibleStates[0] : -1;
        }

        return this.tipPartials[code];
    }

}
