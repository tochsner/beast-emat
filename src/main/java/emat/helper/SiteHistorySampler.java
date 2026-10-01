package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;

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

    // the nodes of the current tree in post-order, so that every child precedes its parent
    final Node[] postOrderNodes;

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

        this.computePostOrder(root);

        for (Node node : this.postOrderNodes) {
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

        int referenceState = this.mutations.getReferenceSequence()[site];

        List<List<Mutation>> siteMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);

            if (node.isRoot()) {
                List<Mutation> rootMutations = new ArrayList<>();
                int rootState = this.nodeStates[nodeNr];
                if (rootState != referenceState) {
                    rootMutations.add(new Mutation(nodeNr, node.getHeight(), node.getHeight(), site, referenceState, rootState));
                }
                siteMutations.add(rootMutations);
            } else {
                siteMutations.add(this.sampleBranchHistory(node, site));
            }
        }

        return siteMutations;
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

    /* Node States */

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
        double logScale = 0.0;

        for (Node node : this.postOrderNodes) {
            double[] partial = this.partials[node.getNr()];

            if (node.isLeaf()) {
                int code = this.alignment.getPattern(this.taxonIndices[node.getNr()], patternIndex);
                System.arraycopy(this.getTipPartial(code), 0, partial, 0, this.numStates);
                continue;
            }

            Arrays.fill(partial, 1.0);

            for (Node child : node.getChildren()) {
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

            logScale += Math.log(maxPartial);
        }

        return logScale;
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

    /**
     * Updates the scale of Q on the branch above the given node and recomputes the
     * transition probabilities P = exp(Q t) of the branch if its scaled length t has changed.
     */
    private void updateTransitionProbabilities(Node node) {
        int nodeNr = node.getNr();

        this.rateScales[nodeNr] = this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
        double duration = this.rateScales[nodeNr] * (node.getParent().getHeight() - node.getHeight());

        if (duration != this.durations[nodeNr]) {
            this.transitionProbabilities[nodeNr] = StochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, duration);
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

    /* Helpers */

    /**
     * Stores the nodes of the tree below the given root in post-order, iteratively so that
     * deep trees cannot overflow the stack.
     */
    private void computePostOrder(Node root) {
        // collect the nodes in pre-order with the children in reverse, then reverse the whole order

        int numVisited = 0;
        Node[] stack = new Node[this.postOrderNodes.length];
        int stackSize = 0;
        stack[stackSize++] = root;

        while (stackSize > 0) {
            Node node = stack[--stackSize];
            this.postOrderNodes[this.postOrderNodes.length - 1 - numVisited++] = node;
            for (Node child : node.getChildren()) {
                stack[stackSize++] = child;
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
