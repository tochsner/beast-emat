package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the complete mutational history of a random site from its exact conditional " +
        "distribution (stochastic mapping). This is a Gibbs move, so it is always accepted.")
public class SiteHistoryGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    // safeguard against numerical problems when sampling the number of uniformised jumps
    private static final int MAX_JUMPS = 10000;

    Mutations mutations;
    GeneticPrior geneticPrior;
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
        this.nodeStates[root.getNr()] = this.sampleIndex(rootWeights);

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

            this.nodeStates[child.getNr()] = this.sampleIndex(weights);
            this.sampleNodeStates(child);
        }
    }

    /**
     * Computes the transition probabilities P = exp(Q t) of the branch above the given node,
     * where t is the branch length scaled by the branch rate and the site rate.
     */
    private void computeTransitionProbabilities(Node node) {
        double[] eigenVectors = this.eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = this.eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = this.eigenDecomposition.getEigenValues();

        double duration = this.getRateScale(node) * (node.getParent().getHeight() - node.getHeight());

        double[] expEigenValues = new double[this.numStates];
        for (int k = 0; k < this.numStates; k++) {
            expEigenValues[k] = Math.exp(eigenValues[k] * duration);
        }

        double[] probabilities = this.transitionProbabilities[node.getNr()];
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double probability = 0.0;
                for (int k = 0; k < this.numStates; k++) {
                    probability += eigenVectors[from * this.numStates + k] * expEigenValues[k]
                            * inverseEigenVectors[k * this.numStates + to];
                }

                // clamp tiny negative values caused by rounding
                probabilities[from * this.numStates + to] = Math.max(probability, 0.0);
            }
        }
    }

    /* Branch Histories */

    /**
     * Samples the history of the given site on the branch above the given node, given the
     * sampled states at both ends. Uses uniformisation: with a rate μ* at least as large as
     * every escape rate, the path is a Poisson number of jumps of the chain B = I + R/μ*,
     * where R is the rate matrix of the branch, and jumps that stay in the same state are
     * dropped. The number of jumps is drawn conditional on the end state, then the jump
     * times uniformly, and then the states of the jumps one by one.
     */
    private List<Mutation> sampleBranchHistory(Node node, int site) {
        Node parent = node.getParent();
        int startState = this.nodeStates[parent.getNr()];
        int endState = this.nodeStates[node.getNr()];

        // evolution runs forwards in time from the parent to the node

        double branchStartHeight = parent.getHeight();
        double branchEndHeight = node.getHeight();
        double duration = branchStartHeight - branchEndHeight;

        List<Mutation> siteMutations = new ArrayList<>();
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return siteMutations;
        }

        // set up the uniformised chain for the rate matrix R of this branch

        double rateScale = this.getRateScale(node);

        double maxEscapeRate = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            maxEscapeRate = Math.max(maxEscapeRate, -rateScale * this.rateMatrix[state * this.numStates + state]);
        }
        if (maxEscapeRate == 0.0) {
            return siteMutations;
        }

        double[] jumpMatrix = new double[this.numStates * this.numStates];
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double identity = from == to ? 1.0 : 0.0;
                jumpMatrix[from * this.numStates + to] = identity + rateScale * this.rateMatrix[from * this.numStates + to] / maxEscapeRate;
            }
        }

        // sample the number of jumps n with P(n | start, end) ∝ Poisson(n; μ* t) (B^n)_{start, end}

        double endProbability = this.transitionProbabilities[node.getNr()][startState * this.numStates + endState];
        double threshold = Randomizer.nextDouble() * endProbability;

        List<double[]> jumpMatrixPowers = new ArrayList<>();
        jumpMatrixPowers.add(this.getIdentityMatrix());

        double poissonProbability = Math.exp(-maxEscapeRate * duration);
        double cumulativeProbability = poissonProbability * jumpMatrixPowers.get(0)[startState * this.numStates + endState];
        int numJumps = 0;

        while (cumulativeProbability < threshold) {
            numJumps++;
            if (numJumps > MAX_JUMPS) {
                throw new RuntimeException("Could not sample the number of jumps on a branch of the site history.");
            }

            poissonProbability *= maxEscapeRate * duration / numJumps;
            double[] jumpMatrixPower = this.multiplyMatrices(jumpMatrixPowers.get(numJumps - 1), jumpMatrix);
            jumpMatrixPowers.add(jumpMatrixPower);

            cumulativeProbability += poissonProbability * jumpMatrixPower[startState * this.numStates + endState];
        }

        // sample the jump times forwards from the branch start

        double[] jumpTimes = new double[numJumps];
        for (int i = 0; i < numJumps; i++) {
            jumpTimes[i] = Randomizer.nextDouble() * duration;
        }
        Arrays.sort(jumpTimes);

        // sample the state after every jump, conditional on reaching the end state, and keep the real jumps

        int state = startState;
        double previousHeight = branchStartHeight;
        double[] weights = new double[this.numStates];

        for (int i = 0; i < numJumps; i++) {
            double[] remainingPower = jumpMatrixPowers.get(numJumps - i - 1);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                weights[nextState] = jumpMatrix[state * this.numStates + nextState]
                        * remainingPower[nextState * this.numStates + endState];
            }
            int nextState = this.sampleIndex(weights);

            if (nextState != state) {
                double height = branchStartHeight - jumpTimes[i];
                siteMutations.add(new Mutation(node.getNr(), height, previousHeight, site, state, nextState));
                previousHeight = height;
                state = nextState;
            }
        }

        return siteMutations;
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

    /** Samples an index with probability proportional to the given non-negative weights. */
    private int sampleIndex(double[] weights) {
        double totalWeight = 0.0;
        for (double weight : weights) {
            totalWeight += weight;
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        double cumulativeWeight = 0.0;
        for (int i = 0; i < weights.length; i++) {
            cumulativeWeight += weights[i];
            if (threshold < cumulativeWeight) {
                return i;
            }
        }

        // only reached through rounding, so return the last index with positive weight
        for (int i = weights.length - 1; i >= 0; i--) {
            if (weights[i] > 0.0) {
                return i;
            }
        }
        throw new RuntimeException("Cannot sample from weights that are all zero.");
    }

    private double[] getIdentityMatrix() {
        double[] identity = new double[this.numStates * this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            identity[state * this.numStates + state] = 1.0;
        }
        return identity;
    }

    private double[] multiplyMatrices(double[] left, double[] right) {
        double[] product = new double[this.numStates * this.numStates];
        for (int i = 0; i < this.numStates; i++) {
            for (int k = 0; k < this.numStates; k++) {
                double value = left[i * this.numStates + k];
                if (value == 0.0) {
                    continue;
                }
                for (int j = 0; j < this.numStates; j++) {
                    product[i * this.numStates + j] += value * right[k * this.numStates + j];
                }
            }
        }
        return product;
    }

}
