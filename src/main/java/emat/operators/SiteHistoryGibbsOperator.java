package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.helper.StochasticMapping;
import emat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the complete mutational history of a random site from its exact conditional " +
        "distribution (stochastic mapping). This is a Gibbs move, so it is always accepted.")
public class SiteHistoryGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    StochasticMapping stochasticMapping;
    TreeInterface tree;
    Alignment alignment;
    DataType dataType;

    int numStates;

    // the model of the current proposal: Q, its eigen decomposition, and the site rate
    double[] rateMatrix;
    EigenDecomposition eigenDecomposition;
    double siteRate;

    // per node: the transition probabilities of the branch above it, the partial
    // likelihoods of the subtree below it, and its sampled state
    double[][] transitionProbabilities;
    double[][] partials;
    int[] nodeStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.mutations.getTree();
        this.alignment = this.mutations.getAlignment();
        this.dataType = this.alignment.getDataType();

        this.numStates = this.alignment.getMaxStateCount();

        int numNodes = this.tree.getNodeCount();
        this.transitionProbabilities = new double[numNodes][this.numStates * this.numStates];
        this.partials = new double[numNodes][this.numStates];
        this.nodeStates = new int[numNodes];

        this.stochasticMapping = new StochasticMapping(this.numStates);
    }

    /**
     * Picks a site uniformly and replaces its history by a draw from the conditional
     * distribution of the genetic prior given the tip data and everything else. Per site,
     * the genetic prior is the density of a CTMC path on the tree, so the draw consists of
     * sampling the node states by pruning and then the path on every branch given its end
     * states. As a Gibbs move, it is always accepted.
     */
    @Override
    public double proposal() {
        int site = Randomizer.nextInt(this.alignment.getSiteCount());
        Node root = this.tree.getRoot();

        // take the current model from the genetic prior

        this.rateMatrix = this.geneticPrior.computeRateMatrix();
        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);
        this.siteRate = this.geneticPrior.siteModel.getRateForCategory(0, root);

        // sample the node states: prune upwards, then sample downwards starting at the root

        this.computePartials(root, site);

        double[] frequencies = this.geneticPrior.substitutionModel.getFrequencies();
        double[] rootWeights = new double[this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            rootWeights[state] = frequencies[state] * this.partials[root.getNr()][state];
        }
        this.nodeStates[root.getNr()] = StochasticMapping.sampleIndex(rootWeights);

        this.sampleNodeStates(root);

        // replace the mutations at the site on every branch

        int referenceState = this.mutations.getReferenceSequence()[site];

        for (Node node : this.tree.getNodesAsArray()) {
            List<Mutation> newSiteMutations;

            if (node.isRoot()) {
                newSiteMutations = new ArrayList<>();
                int rootState = this.nodeStates[node.getNr()];
                if (rootState != referenceState) {
                    newSiteMutations.add(new Mutation(node.getNr(), node.getHeight(), node.getHeight(), site, referenceState, rootState));
                }
            } else {
                newSiteMutations = this.sampleBranchHistory(node, site);
            }

            this.replaceSiteMutations(node, site, newSiteMutations);
        }

        return Double.POSITIVE_INFINITY;
    }

    /* Node States */

    /**
     * Computes the partial likelihoods of the given node and of every node below it at the
     * given site, together with the transition probabilities of the branches below it. A tip
     * allows every state compatible with its observed character. The partials are rescaled,
     * which does not change the sampled states.
     */
    private void computePartials(Node node, int site) {
        double[] partial = this.partials[node.getNr()];

        if (node.isLeaf()) {
            int taxonNr = this.alignment.getTaxonIndex(node.getID());
            int code = this.alignment.getPattern(taxonNr, this.alignment.getPatternIndex(site));

            Arrays.fill(partial, 0.0);
            for (int state : this.dataType.getStatesForCode(code)) {
                partial[state] = 1.0;
            }

            return;
        }

        Arrays.fill(partial, 1.0);

        for (Node child : node.getChildren()) {
            this.computePartials(child, site);
            this.computeTransitionProbabilities(child);

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

    /**
     * Replaces the mutations at the given site on the branch above the given node, keeping
     * the mutations at all other sites. Branches without mutations at the site before and
     * after are left untouched.
     */
    private void replaceSiteMutations(Node node, int site, List<Mutation> newSiteMutations) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);

        List<Mutation> newBranchMutations = new ArrayList<>();
        for (Mutation mutation : branchMutations) {
            if (mutation.site() != site) {
                newBranchMutations.add(mutation);
            }
        }

        if (newBranchMutations.size() == branchMutations.size() && newSiteMutations.isEmpty()) {
            return;
        }

        newBranchMutations.addAll(newSiteMutations);
        newBranchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        this.mutations.applyMutations(node, newBranchMutations, this);
    }

    /* Helpers */

    /** Returns the factor that scales Q on the branch above the given node: the branch rate times the site rate. */
    private double getRateScale(Node node) {
        return this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
    }

}
