package esat.prior;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.CalculationNode;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.GenericTreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import emat.tree.IncrementalStoreTree;
import esat.helper.TipCompatibility;
import esat.state.NodeStates;

import java.util.Arrays;

/**
 * The genetic prior of explicit node states, with the mutation history of every branch
 * integrated out. The root contributes the log frequency of its state at every site, and
 * every branch contributes the log transition probability from the state of its parent to the
 * state of its node at every site. Both only depend on the root state counts and the
 * transition counts of the states, so a branch costs O(K^2) for K states, independent of the
 * number of sites and mutations.
 * <p>
 * The states of the tips at missing and ambiguous sites are latent, and the prior is zero if
 * a tip is in a state its data is not compatible with. Summing the prior over the compatible
 * latent states gives the tree likelihood.
 */
@Description("Implements the genetic prior for explicit node states. Site rate categories are not yet supported.")
public class GeneticPrior extends GenericTreeLikelihood {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "", Input.Validate.REQUIRED);

    // the running total drifts by rounding, so it is re-summed from the branches after this many incremental updates
    private static final int RESUM_INTERVAL = 1000;

    /** The coefficients 1/n! of x^(n-2) in the Taylor series of (exp(x) - 1 - x) / x^2. */
    private static final double[] TAYLOR_COEFFICIENTS = {
            1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0, 1.0 / 40320.0,
            1.0 / 362880.0, 1.0 / 3628800.0, 1.0 / 39916800.0, 1.0 / 479001600.0, 1.0 / 6227020800.0, 1.0 / 87178291200.0
    };

    TreeInterface tree;
    Alignment alignment;
    NodeStates nodeStates;

    public SiteModel.Base siteModel;
    public SubstitutionModel substitutionModel;
    public Base branchRateModel;

    int numStates;

    TipCompatibility tipCompatibility;

    // the eigen decomposition of the rate matrix Q = V diag(λ) V^-1, the rate of the single site category, and the log frequencies, as of the last calculation
    private double[] eigenVectors;
    private double[] inverseEigenVectors;
    private double[] eigenValues;
    private double categoryRate;
    private double[] frequencies;
    private double[] logFrequencies;

    // the log contribution of the branch above each node, which is zero for the root
    private double[] branchLogPs;
    private double[] storedBranchLogPs;

    // the log contribution of the root sequence
    private double rootLogP;
    private double storedRootLogP;

    // the running sum of the contributions of the root and of every branch
    private double summedLogP;
    private double storedSummedLogP;

    // whether the states of each tip are compatible with its data, which is true for every inner node
    private boolean[] isTipCompatible;
    private boolean[] storedIsTipCompatible;
    private int numIncompatibleTips;
    private int storedNumIncompatibleTips;

    // whether the caches hold the values of a previous calculation
    private boolean isInitialised = false;

    // the number of incremental updates since the total was last summed from the branches
    private int numIncrementalUpdates = 0;

    // per node, the last calculation in which its branch was dirty
    private int[] dirtyCalculations;
    private int calculationNr = 0;

    // the nodes marked dirty in the current calculation
    private int[] dirtyNodeNrs;
    private int numDirtyNodes = 0;

    // scratch space for the transition probabilities of a branch and the exponentials of the eigenvalues
    private double[] transitionProbabilities;
    private double[] scaledExponentials;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.dataInput.get();
        this.nodeStates = this.nodeStatesInput.get();

        // set up the evolutionary model

        if (!(this.siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        }
        this.siteModel = (SiteModel.Base) this.siteModelInput.get();
        this.siteModel.setDataType(this.alignment.getDataType());
        this.substitutionModel = this.siteModel.substModelInput.get();

        if (this.siteModel.getCategoryCount() != 1) {
            throw new IllegalArgumentException("Site rate categories are not supported yet.");
        }

        if (this.branchRateModelInput.get() != null) {
            this.branchRateModel = this.branchRateModelInput.get();
        } else {
            this.branchRateModel = new StrictClockModel();
        }

        this.numStates = this.alignment.getMaxStateCount();
        if (this.nodeStates.getNumStates() != this.numStates) {
            throw new IllegalArgumentException("The node states have a different number of states than the alignment.");
        }

        this.tipCompatibility = new TipCompatibility(this.alignment, this.tree);

        // set up the caches

        int numNodes = this.tree.getNodeCount();
        this.branchLogPs = new double[numNodes];
        this.storedBranchLogPs = new double[numNodes];

        // every tip counts as compatible until it is checked in the first calculation
        this.isTipCompatible = new boolean[numNodes];
        this.storedIsTipCompatible = new boolean[numNodes];
        Arrays.fill(this.isTipCompatible, true);

        this.dirtyCalculations = new int[numNodes];
        this.dirtyNodeNrs = new int[numNodes];

        this.transitionProbabilities = new double[this.numStates * this.numStates];
        this.scaledExponentials = new double[this.numStates];
    }

    /**
     * Computes the genetic prior and caches it as logP. Only the branches whose contribution
     * may have changed since the previous call are recomputed: those whose node is dirty in
     * the tree or in the node states. Unlike for explicit mutations, the contribution of a
     * branch does not depend on the branches above it. Everything is recomputed on the first
     * call or if the evolutionary model changed.
     */
    @Override
    public double calculateLogP() {
        Node root = this.tree.getRoot();

        // the model may have been restored since the last calculation, so it is fetched every time
        this.updateModel(root);

        boolean isEverythingDirty = !this.isInitialised
                || this.siteModel.somethingIsDirty()
                || (this.substitutionModel instanceof CalculationNode calculationNode && calculationNode.somethingIsDirty())
                || this.branchRateModel.somethingIsDirty();

        // mark the dirty nodes with the current calculation number

        this.calculationNr++;
        this.numDirtyNodes = 0;

        if (!isEverythingDirty && this.tree instanceof IncrementalStoreTree incrementalTree
                && incrementalTree.areFlaggedNodesComplete() && !this.nodeStates.isEveryBranchDirty()) {
            // only the flagged nodes of the tree and the dirty branches of the node states can be dirty
            this.markDirtyNodes(incrementalTree.getFlaggedNodeNrs(), incrementalTree.getFlaggedNodeCount());
            this.markDirtyNodes(this.nodeStates.getDirtyNodeNrs(), this.nodeStates.getDirtyNodeCount());
        } else {
            for (Node node : this.tree.getNodesAsArray()) {
                if (isEverythingDirty || this.isChanged(node)) {
                    this.markDirty(node);
                }
            }
        }

        // update the dirty branches and the root

        for (int i = 0; i < this.numDirtyNodes; i++) {
            this.updateBranch(this.tree.getNode(this.dirtyNodeNrs[i]));
        }
        this.updateTipCompatibilities();

        if (isEverythingDirty || this.nodeStates.isDirty(root)) {
            double oldRootLogP = this.rootLogP;
            this.rootLogP = this.computeRootLogP();
            this.summedLogP += this.rootLogP - oldRootLogP;
        }

        // periodically recompute the entire sum to avoid accumulating rounding errors, and whenever an infinite contribution broke the running sum

        if (isEverythingDirty || ++this.numIncrementalUpdates >= RESUM_INTERVAL || !Double.isFinite(this.summedLogP)) {
            this.numIncrementalUpdates = 0;
            this.summedLogP = this.rootLogP;
            for (double branchLogP : this.branchLogPs) {
                this.summedLogP += branchLogP;
            }
        }

        this.isInitialised = true;

        // the incompatible tips are kept out of the running sum, which could not recover from them
        this.logP = this.numIncompatibleTips > 0 ? Double.NEGATIVE_INFINITY : this.summedLogP;
        return this.logP;
    }

    /**
     * Fetches the eigen decomposition of the rate matrix, the rate of the single category of
     * the site model including the mutation rate, and the log frequencies. The substitution
     * model is shared by all branches, so the decomposition of the root is used.
     */
    private void updateModel(Node root) {
        EigenDecomposition eigenDecomposition = this.substitutionModel.getEigenDecomposition(root);

        double[] imaginaryEigenValues = eigenDecomposition.getImEigenValues();
        if (imaginaryEigenValues != null) {
            for (double imaginaryEigenValue : imaginaryEigenValues) {
                if (imaginaryEigenValue != 0.0) {
                    throw new IllegalArgumentException("Substitution models with complex eigenvalues are not supported.");
                }
            }
        }

        // the arrays are copied, as some substitution models update them in place
        this.eigenVectors = eigenDecomposition.getEigenVectors().clone();
        this.inverseEigenVectors = eigenDecomposition.getInverseEigenVectors().clone();
        this.eigenValues = eigenDecomposition.getEigenValues().clone();

        this.categoryRate = this.siteModel.getRateForCategory(0, root);

        this.frequencies = this.substitutionModel.getFrequencies().clone();
        this.logFrequencies = new double[this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            this.logFrequencies[state] = Math.log(this.frequencies[state]);
        }
    }

    /** Marks those of the given nodes as dirty whose branch changed since the last calculation. */
    private void markDirtyNodes(int[] nodeNrs, int numNodes) {
        for (int i = 0; i < numNodes; i++) {
            Node node = this.tree.getNode(nodeNrs[i]);
            if (this.isChanged(node)) {
                this.markDirty(node);
            }
        }
    }

    private boolean isChanged(Node node) {
        return node.isDirty() != Tree.IS_CLEAN || this.nodeStates.isDirty(node);
    }

    private void markDirty(Node node) {
        if (this.dirtyCalculations[node.getNr()] != this.calculationNr) {
            this.dirtyCalculations[node.getNr()] = this.calculationNr;
            this.dirtyNodeNrs[this.numDirtyNodes++] = node.getNr();
        }
    }

    /** Recomputes the contribution of the branch above the given node and adds its change to the running sum. */
    private void updateBranch(Node node) {
        int nodeNr = node.getNr();

        double oldBranchLogP = this.branchLogPs[nodeNr];
        this.branchLogPs[nodeNr] = node.isRoot() ? 0.0 : this.computeBranchLogP(node);
        this.summedLogP += this.branchLogPs[nodeNr] - oldBranchLogP;
    }

    /** Computes Σ N_ab log P_ab(d) over the transition counts of the branch above the given non-root node. */
    private double computeBranchLogP(Node node) {
        this.getTransitionProbabilities(node, this.transitionProbabilities);
        int[] counts = this.nodeStates.getTransitionCounts(node);

        double branchLogP = 0.0;
        for (int i = 0; i < counts.length; i++) {
            // pairs without sites are skipped, so that impossible transitions do not contribute 0 * log(0)
            if (counts[i] != 0) {
                branchLogP += counts[i] * Math.log(this.transitionProbabilities[i]);
            }
        }
        return branchLogP;
    }

    /**
     * Rechecks the compatibility of the tips whose states may have become incompatible with
     * their data, which is every tip if all states were replaced. The samplers of the states
     * only draw compatible states, so they do not mark any tip.
     */
    private void updateTipCompatibilities() {
        if (!this.isInitialised || this.nodeStates.isEveryCheckPending()) {
            this.nodeStates.visitStates((node, sequence) -> {
                if (node.isLeaf()) {
                    this.updateTipCompatibility(node, sequence);
                }
            });
            return;
        }

        for (int i = 0; i < this.nodeStates.getCheckNodeCount(); i++) {
            Node tip = this.tree.getNode(this.nodeStates.getCheckNodeNrs()[i]);
            this.updateTipCompatibility(tip, this.nodeStates.materializeState(tip));
        }
    }

    private void updateTipCompatibility(Node tip, int[] sequence) {
        boolean wasCompatible = this.isTipCompatible[tip.getNr()];
        boolean isCompatible = this.tipCompatibility.isCompatible(tip, sequence);

        this.isTipCompatible[tip.getNr()] = isCompatible;
        if (wasCompatible && !isCompatible) {
            this.numIncompatibleTips++;
        } else if (!wasCompatible && isCompatible) {
            this.numIncompatibleTips--;
        }
    }

    /** Computes Σ n_a log π_a over the states of the root sequence. */
    private double computeRootLogP() {
        int[] rootStateCounts = this.nodeStates.getRootStateCounts();

        double rootLogP = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            if (rootStateCounts[state] != 0) {
                rootLogP += rootStateCounts[state] * this.logFrequencies[state];
            }
        }
        return rootLogP;
    }

    /**
     * Writes the transition probabilities P(d) = exp(Q d) of the branch above the given
     * non-root node into the given array, row-major by the state at the parent, using the
     * model as of the last calculation or refresh. They are computed as I + V diag(expm1(λ d)) V^-1,
     * which keeps the small transition probabilities of short branches accurate.
     */
    public void getTransitionProbabilities(Node node, double[] transitionProbabilities) {
        this.getTransitionProbabilities(node, node.getParent().getHeight() - node.getHeight(), transitionProbabilities);
    }

    /**
     * Writes the transition probabilities of a branch of the given length into the given
     * array, at the rate of the branch above the given node, as above. This evaluates
     * branches that a proposal considers but that are not in the tree.
     */
    public void getTransitionProbabilities(Node node, double branchLength, double[] transitionProbabilities) {
        double distance = this.categoryRate * this.branchRateModel.getRateForBranch(node) * branchLength;

        for (int k = 0; k < this.numStates; k++) {
            this.scaledExponentials[k] = computeExpm1(this.eigenValues[k] * distance);
        }

        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double probability = from == to ? 1.0 : 0.0;
                for (int k = 0; k < this.numStates; k++) {
                    probability += this.eigenVectors[from * this.numStates + k] * this.scaledExponentials[k]
                            * this.inverseEigenVectors[k * this.numStates + to];
                }
                transitionProbabilities[from * this.numStates + to] = probability;
            }
        }
    }

    /**
     * Computes exp(x) - 1 without cancellation. Math.expm1 is accurate too, but several times
     * slower than Math.exp, which the JVM intrinsifies. Near zero, the Taylor series is accurate
     * to machine precision after a few terms, and elsewhere the subtraction loses nothing.
     */
    private static double computeExpm1(double x) {
        if (Math.abs(x) >= 0.5) {
            return Math.exp(x) - 1.0;
        }

        double sum = TAYLOR_COEFFICIENTS[TAYLOR_COEFFICIENTS.length - 1];
        for (int order = TAYLOR_COEFFICIENTS.length - 2; order >= 0; order--) {
            sum = TAYLOR_COEFFICIENTS[order] + sum * x;
        }
        return x * (1.0 + x * sum);
    }

    /**
     * Fetches the current evolutionary model, which the transition probabilities and the
     * frequencies then use. Samplers call this before they use the model, as it may have
     * changed since the last calculation.
     */
    public void refreshModel() {
        this.updateModel(this.tree.getRoot());
    }

    /** Returns the root frequencies of the model as of the last calculation or refresh. */
    public double[] getFrequencies() {
        return this.frequencies;
    }

    /** Returns the constraints of the data on the latent states of the tips. */
    public TipCompatibility getTipCompatibility() {
        return this.tipCompatibility;
    }

    /* CalculationNode methods */

    @Override
    public void store() {
        System.arraycopy(this.branchLogPs, 0, this.storedBranchLogPs, 0, this.branchLogPs.length);
        System.arraycopy(this.isTipCompatible, 0, this.storedIsTipCompatible, 0, this.isTipCompatible.length);
        this.storedRootLogP = this.rootLogP;
        this.storedSummedLogP = this.summedLogP;
        this.storedNumIncompatibleTips = this.numIncompatibleTips;
        super.store();
    }

    @Override
    public void restore() {
        double[] branchLogPs = this.branchLogPs;
        this.branchLogPs = this.storedBranchLogPs;
        this.storedBranchLogPs = branchLogPs;

        boolean[] isTipCompatible = this.isTipCompatible;
        this.isTipCompatible = this.storedIsTipCompatible;
        this.storedIsTipCompatible = isTipCompatible;

        this.rootLogP = this.storedRootLogP;
        this.summedLogP = this.storedSummedLogP;
        this.numIncompatibleTips = this.storedNumIncompatibleTips;

        super.restore();
    }

}
