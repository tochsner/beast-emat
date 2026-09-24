package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Samples the complete history of a single site on the tree from its exact conditional
 * distribution under the model of the genetic prior, given the tip data (stochastic
 * mapping). Per site, the genetic prior is the density of a CTMC path on the tree, so a draw
 * consists of sampling the node states by pruning and then the path on every branch given
 * its end states. The probability of a history is its genetic prior divided by the
 * likelihood of the tip data, which computeLogLikelihood provides.
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

    // per node: the transition probabilities of the branch above it, the partial
    // likelihoods of the subtree below it, and its sampled state
    final double[][] transitionProbabilities;
    final double[][] partials;
    final int[] nodeStates;

    public SiteHistorySampler(Mutations mutations, GeneticPrior geneticPrior) {
        this.mutations = mutations;
        this.geneticPrior = geneticPrior;
        this.tree = mutations.getTree();
        this.alignment = mutations.getAlignment();
        this.dataType = this.alignment.getDataType();

        this.numStates = this.alignment.getMaxStateCount();
        this.stochasticMapping = new StochasticMapping(this.numStates);

        int numNodes = this.tree.getNodeCount();
        this.transitionProbabilities = new double[numNodes][this.numStates * this.numStates];
        this.partials = new double[numNodes][this.numStates];
        this.nodeStates = new int[numNodes];
    }

    /**
     * Takes the current model from the genetic prior and computes the transition
     * probabilities of every branch of the current tree. This has to be called again
     * whenever the tree or the model changes.
     */
    public void updateModel() {
        Node root = this.tree.getRoot();

        this.rateMatrix = this.geneticPrior.computeRateMatrix();
        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);
        this.siteRate = this.geneticPrior.siteModel.getRateForCategory(0, root);
        this.frequencies = this.geneticPrior.substitutionModel.getFrequencies();

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                this.computeTransitionProbabilities(node);
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

        this.computePartials(root, this.alignment.getPatternIndex(site));

        double[] rootWeights = new double[this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            rootWeights[state] = this.frequencies[state] * this.partials[root.getNr()][state];
        }
        this.nodeStates[root.getNr()] = StochasticMapping.sampleIndex(rootWeights);

        this.sampleNodeStates(root);

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

    /* Node States */

    /** Computes the log likelihood of the tip data at a single site with the given pattern. */
    private double computePatternLogLikelihood(int patternIndex) {
        Node root = this.tree.getRoot();
        double logScale = this.computePartials(root, patternIndex);

        double likelihood = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            likelihood += this.frequencies[state] * this.partials[root.getNr()][state];
        }

        return logScale + Math.log(likelihood);
    }

    /**
     * Computes the partial likelihoods of the given node and of every node below it at a
     * site with the given pattern. A tip allows every state compatible with its observed
     * character. The partials are rescaled, which does not change the sampled states, and
     * the log of all rescaling factors below and at the node is returned.
     */
    private double computePartials(Node node, int patternIndex) {
        double[] partial = this.partials[node.getNr()];

        if (node.isLeaf()) {
            int taxonNr = this.alignment.getTaxonIndex(node.getID());
            int code = this.alignment.getPattern(taxonNr, patternIndex);

            Arrays.fill(partial, 0.0);
            for (int state : this.dataType.getStatesForCode(code)) {
                partial[state] = 1.0;
            }

            return 0.0;
        }

        Arrays.fill(partial, 1.0);
        double logScale = 0.0;

        for (Node child : node.getChildren()) {
            logScale += this.computePartials(child, patternIndex);

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

        return logScale + Math.log(maxPartial);
    }

    /**
     * Samples the states of every node below the given node, given the state of the node.
     * A child state is drawn proportional to the transition probability from the parent
     * state times the partial likelihood of the child.
     */
    private void sampleNodeStates(Node node) {
        int parentState = this.nodeStates[node.getNr()];

        for (Node child : node.getChildren()) {
            double[] childPartial = this.partials[child.getNr()];
            double[] childTransitionProbabilities = this.transitionProbabilities[child.getNr()];

            double[] weights = new double[this.numStates];
            for (int childState = 0; childState < this.numStates; childState++) {
                weights[childState] = childTransitionProbabilities[parentState * this.numStates + childState] * childPartial[childState];
            }

            this.nodeStates[child.getNr()] = StochasticMapping.sampleIndex(weights);
            this.sampleNodeStates(child);
        }
    }

    /**
     * Computes the transition probabilities P = exp(Q t) of the branch above the given node,
     * where t is the branch length scaled by the branch rate and the site rate.
     */
    private void computeTransitionProbabilities(Node node) {
        double duration = this.getRateScale(node) * (node.getParent().getHeight() - node.getHeight());
        this.transitionProbabilities[node.getNr()] = this.stochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, duration);
    }

    /* Branch Histories */

    /**
     * Samples the history of the given site on the branch above the given node, given the
     * sampled states at both ends. The rate matrix of the branch is Q scaled by the branch
     * rate and the site rate.
     */
    private List<Mutation> sampleBranchHistory(Node node, int site) {
        Node parent = node.getParent();
        int startState = this.nodeStates[parent.getNr()];
        int endState = this.nodeStates[node.getNr()];

        double rateScale = this.getRateScale(node);
        double[] branchRateMatrix = new double[this.numStates * this.numStates];
        for (int i = 0; i < branchRateMatrix.length; i++) {
            branchRateMatrix[i] = rateScale * this.rateMatrix[i];
        }

        double endProbability = this.transitionProbabilities[node.getNr()][startState * this.numStates + endState];

        return this.stochasticMapping.sampleBranchHistory(
                node.getNr(), site, parent.getHeight(), node.getHeight(),
                startState, endState, branchRateMatrix, endProbability
        );
    }

    /* Helpers */

    /** Returns the factor that scales Q on the branch above the given node: the branch rate times the site rate. */
    private double getRateScale(Node node) {
        return this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
    }

}
