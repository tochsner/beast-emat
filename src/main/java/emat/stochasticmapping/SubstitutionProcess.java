package emat.stochasticmapping;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.spec.inference.parameter.IntVectorParam;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;

import java.util.Arrays;

/**
 * The substitution process of the genetic prior as of the last update: the rate matrix Q, its
 * eigen decomposition, the rates of the site categories, the category of every site and the
 * root frequencies. The samplers of mutation histories read the model from here, so that it is
 * fetched once per proposal. On the branch above a node, a site of category c runs with the rate
 * matrix s r_c Q, where the branch rate s is the branch rate and r_c the rate of its category.
 */
public final class SubstitutionProcess {

    private final GeneticPrior geneticPrior;
    private final TreeInterface tree;
    private final int numStates;
    private final int numCategories;

    // the model as of the last update; Q is only replaced by a new array if its values change
    private double[] rateMatrix;
    private EigenDecomposition eigenDecomposition;
    private double[] categoryRates;
    private double[] frequencies;
    private double maxEscapeRate;

    // the category of every site as of the last update, and the sites of every category in increasing order
    private final int[] siteCategories;
    private long comparedSiteCategoriesVersion = -1;
    private boolean isSitesOfCategoryStale = true;
    private int[][] sitesOfCategory;

    public SubstitutionProcess(GeneticPrior geneticPrior, TreeInterface tree, int numStates) {
        this.geneticPrior = geneticPrior;
        this.tree = tree;
        this.numStates = numStates;
        this.numCategories = geneticPrior.getNumCategories();
        this.categoryRates = new double[this.numCategories];
        this.siteCategories = new int[geneticPrior.getNumSites()];
    }

    /**
     * Takes the current model from the genetic prior, which has to be done once per proposal.
     * The substitution model may overwrite its eigen decomposition objects in place, so they
     * are fetched again every time. Q keeps its array while its values stay the same, so
     * users can detect a change by comparing the array they saw last. The categories of the
     * sites are compared with the last update, which only costs time when they change.
     */
    public void update() {
        Node root = this.tree.getRoot();

        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);
        double[] newRateMatrix = this.geneticPrior.computeRateMatrix();
        if (!Arrays.equals(newRateMatrix, this.rateMatrix)) {
            this.rateMatrix = newRateMatrix;
            this.maxEscapeRate = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                this.maxEscapeRate = Math.max(this.maxEscapeRate, -this.rateMatrix[state * this.numStates + state]);
            }
        }

        for (int category = 0; category < this.numCategories; category++) {
            this.categoryRates[category] = this.geneticPrior.siteModel.getRateForCategory(category, root);
        }
        this.frequencies = this.geneticPrior.substitutionModel.getFrequencies();

        this.updateSiteCategories();
    }

    /** Copies the categories of the sites from the prior, and rebuilds the sites of every category if any of them changed. */
    private void updateSiteCategories() {
        IntVectorParam<?> categories = this.geneticPrior.getSiteRateCategories();
        if (categories == null || this.comparedSiteCategoriesVersion == this.geneticPrior.getSiteCategoriesVersion()) {
            return;
        }
        this.comparedSiteCategoriesVersion = this.geneticPrior.getSiteCategoriesVersion();

        boolean isChanged = false;
        for (int site = 0; site < this.siteCategories.length; site++) {
            int category = categories.get(site);
            if (category != this.siteCategories[site]) {
                this.siteCategories[site] = category;
                isChanged = true;
            }
        }

        if (isChanged) {
            this.isSitesOfCategoryStale = true;
        }
    }

    /** Lists the sites of every category in increasing order. */
    private void rebuildSitesOfCategory() {
        this.isSitesOfCategoryStale = false;
        int[] counts = new int[this.numCategories];
        for (int category : this.siteCategories) {
            counts[category]++;
        }

        this.sitesOfCategory = new int[this.numCategories][];
        for (int category = 0; category < this.numCategories; category++) {
            this.sitesOfCategory[category] = new int[counts[category]];
        }

        int[] fillCounts = new int[this.numCategories];
        for (int site = 0; site < this.siteCategories.length; site++) {
            int category = this.siteCategories[site];
            this.sitesOfCategory[category][fillCounts[category]++] = site;
        }
    }

    public int getNumStates() {
        return this.numStates;
    }

    public int getNumCategories() {
        return this.numCategories;
    }

    /** Returns the category of the given site as of the last update. */
    public int getCategory(int site) {
        return this.siteCategories[site];
    }

    /** Returns the sites of the given category in increasing order as of the last update. */
    public int[] getSitesOfCategory(int category) {
        if (this.isSitesOfCategoryStale) {
            this.rebuildSitesOfCategory();
        }
        return this.sitesOfCategory[category];
    }

    /** Returns Q, stored row by row, before the scaling of a branch and a category. */
    public double[] getRateMatrix() {
        return this.rateMatrix;
    }

    /** Returns the rate of the given category, including the mutation rate of the site model. */
    public double getCategoryRate(int category) {
        return this.categoryRates[category];
    }

    public double[] getFrequencies() {
        return this.frequencies;
    }

    /** Returns the branch rate above the given node, which scales every category alike. */
    public double getBranchRate(Node node) {
        return this.geneticPrior.branchRateModel.getRateForBranch(node);
    }

    /** Returns the scale of Q on the branch above the given node for sites of the given category: the branch rate times the category rate. */
    public double getRateScale(Node node, int category) {
        return this.getBranchRate(node) * this.categoryRates[category];
    }

    /**
     * Computes the transition probabilities P = exp(Q t) into the given array of length n²,
     * where the duration t already includes every rate scaling of the branch.
     */
    public void computeTransitionProbabilities(double duration, double[] probabilities) {
        computeTransitionProbabilities(this.eigenDecomposition, duration, probabilities);
    }

    /* Uniformisation */

    /**
     * Returns the rate of the uniformised process: the largest escape rate of Q, which bounds
     * the escape rate of every state. Events of a Poisson process at this rate, of which each
     * is a jump of the chain with probability escape rate over this rate and otherwise
     * virtual, yield the same process as Q.
     */
    public double getMaxEscapeRate() {
        return this.maxEscapeRate;
    }

    /**
     * Resolves an event of the uniformised process in the given state: draws a threshold
     * uniformly up to the rate of the uniformised process, and returns the state whose
     * off-diagonal rate covers it, or the given state if the threshold lies beyond the escape
     * rate and the event is virtual.
     */
    public int sampleUniformisedJump(int state) {
        double threshold = Randomizer.nextDouble() * this.maxEscapeRate;

        double cumulativeRate = 0.0;
        for (int newState = 0; newState < this.numStates; newState++) {
            if (newState == state) {
                continue;
            }

            cumulativeRate += this.rateMatrix[state * this.numStates + newState];
            if (threshold < cumulativeRate) {
                return newState;
            }
        }

        return state;
    }

    /* Transition Probabilities */

    /**
     * Computes the transition probabilities P = exp(Q t) from the eigen decomposition of Q,
     * where the duration t already includes every rate scaling of the branch.
     */
    public static double[] computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration) {
        int numStates = eigenDecomposition.getEigenValues().length;
        double[] probabilities = new double[numStates * numStates];
        computeTransitionProbabilities(eigenDecomposition, duration, probabilities);
        return probabilities;
    }

    /**
     * Computes the transition probabilities P = exp(Q t) like computeTransitionProbabilities,
     * but writes them into the given array of length n², which avoids an allocation per branch.
     */
    public static void computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration, double[] probabilities) {
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();
        int numStates = eigenValues.length;

        // the eigenvector products sum to the identity, so exp(λ t) = 1 + expm1(λ t) gives P = I + Σ_k V_k W_k expm1(λ_k t);
        // this avoids the cancellation of terms of size one, which dominates for short times
        Arrays.fill(probabilities, 0.0);
        for (int state = 0; state < numStates; state++) {
            probabilities[state * numStates + state] = 1.0;
        }
        for (int k = 0; k < numStates; k++) {
            // a zero eigenvalue, which every rate matrix has, contributes nothing to the correction
            double expm1EigenValue = eigenValues[k] == 0.0 ? 0.0 : Math.expm1(eigenValues[k] * duration);

            for (int from = 0; from < numStates; from++) {
                double scaledEigenVector = eigenVectors[from * numStates + k] * expm1EigenValue;
                for (int to = 0; to < numStates; to++) {
                    probabilities[from * numStates + to] += scaledEigenVector * inverseEigenVectors[k * numStates + to];
                }
            }
        }

        // clamp tiny negative values caused by rounding
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = Math.max(probabilities[i], 0.0);
        }
    }

}
