package emat.operators.categories;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.SubstitutionProcess;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * The posterior of the rate category of every site given the tree and the mutations. Given the
 * histories, the category c of a site with weight w_c in the site model has the probability
 * proportional to w_c r_c^M exp(-r_c L), where r_c is the rate of the category, M the number of
 * mutations of the site on the non-root branches, and L the rate-weighted time that the site
 * spends in its escape rates. Most sites have no mutations, and their posterior only depends on
 * the state at the root, so it is computed once per root state and shared by those sites.
 */
public class SiteCategoryPosterior {

    private final Mutations mutations;
    private final TreeInterface tree;
    private final SubstitutionProcess process;
    private final int numStates;
    private final int numCategories;

    // per site: the number of mutations on the non-root branches, the rate-weighted time of its escape rates, and its state at the root
    private final int[] numMutations;
    private final double[] escapeTimes;
    private final int[] rootStates;

    // the rate-weighted length of the whole tree, and the proportions of the categories as of the last update
    private double totalLength;
    private double[] categoryProportions;

    // the posterior of the sites without mutations, by their root state: the normalised probabilities and their logs
    private final double[][] invariantProbabilities;
    private final double[][] invariantLogProbabilities;

    public SiteCategoryPosterior(Mutations mutations, TreeInterface tree, SubstitutionProcess process) {
        this.mutations = mutations;
        this.tree = tree;
        this.process = process;
        this.numStates = process.getNumStates();
        this.numCategories = process.getNumCategories();

        int numSites = mutations.getReferenceSequence().length;
        this.numMutations = new int[numSites];
        this.escapeTimes = new double[numSites];
        this.rootStates = new int[numSites];

        this.categoryProportions = new double[this.numCategories];
        this.invariantProbabilities = new double[this.numStates][this.numCategories];
        this.invariantLogProbabilities = new double[this.numStates][this.numCategories];
    }

    /**
     * Sets the proportions of the categories, and computes the posterior of the sites without
     * mutations under them. The statistics have to be collected before, and the process has to be
     * updated, as the category rates are read from it.
     */
    public void setCategoryProportions(double[] categoryProportions) {
        System.arraycopy(categoryProportions, 0, this.categoryProportions, 0, this.numCategories);
        this.computeInvariantPosteriors();
    }

    /** Returns the number of categories of the site model. */
    public int getNumCategories() {
        return this.numCategories;
    }

    /** Writes the normalised probabilities of the categories of the given site into the given array. */
    public void computeProbabilities(int site, double[] probabilities) {
        if (this.numMutations[site] == 0) {
            System.arraycopy(this.invariantProbabilities[this.rootStates[site]], 0, probabilities, 0, this.numCategories);
            return;
        }

        this.computeLogProbabilities(site, probabilities);
        normaliseLogProbabilities(probabilities, probabilities);
    }

    /** Returns the normalised log probability of the given category of the given site. */
    public double computeLogProbability(int site, int category) {
        if (this.numMutations[site] == 0) {
            return this.invariantLogProbabilities[this.rootStates[site]][category];
        }

        double[] logProbabilities = new double[this.numCategories];
        this.computeLogProbabilities(site, logProbabilities);
        return logProbabilities[category];
    }

    /* Site Statistics */

    /**
     * Collects for every site the number of its mutations on the non-root branches and the
     * rate-weighted time L of its escape rates. The escape rate of a state is the one of the
     * unscaled rate matrix. With the states x_s(t) of a site, L is the integral over the tree of
     * the escape rate of x_s(t), weighted by the branch rates. It starts at the root state
     * times the total weighted length, and every non-root mutation changes the escape rate for
     * the part of the tree below it, which is the weighted length of its branch below the
     * mutation plus the weighted length below the branch. The statistics do not depend on the
     * shape or the categories, so they only have to be collected once per change of the tree or
     * the histories. The process has to be updated before, as the rate matrix is read from it.
     */
    public void collectSiteStatistics() {
        Node root = this.tree.getRoot();
        double[] subtreeLengths = this.computeSubtreeLengths();
        this.computeRootStates();

        Arrays.fill(this.numMutations, 0);
        this.totalLength = subtreeLengths[root.getNr()];
        for (int site = 0; site < this.escapeTimes.length; site++) {
            this.escapeTimes[site] = this.getEscapeRate(this.rootStates[site]) * this.totalLength;
        }

        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            double branchRate = this.process.getBranchRate(node);
            List<Mutation> nodeMutations = this.mutations.getMutations(node);
            for (int i = 0; i < nodeMutations.size(); i++) {
                Mutation mutation = nodeMutations.get(i);
                int site = mutation.site();
                double escapeRateChange = this.getEscapeRate(mutation.newState()) - this.getEscapeRate(mutation.oldState());

                this.numMutations[site]++;
                this.escapeTimes[site] += escapeRateChange
                        * (branchRate * (mutation.time() - node.getHeight()) + subtreeLengths[node.getNr()]);
            }
        }
    }

    /**
     * Computes for every node the rate-weighted length of the branches below it, which is the sum
     * of the branch rate times the height difference over all branches of its subtree. The
     * nodes are visited with the children before their parents.
     */
    private double[] computeSubtreeLengths() {
        Node[] preOrder = this.computePreOrder();
        double[] subtreeLengths = new double[this.tree.getNodeCount()];

        for (int i = preOrder.length - 1; i >= 0; i--) {
            Node node = preOrder[i];
            double length = 0.0;
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                Node child = node.getChild(childIndex);
                length += subtreeLengths[child.getNr()] + this.process.getBranchRate(child) * (node.getHeight() - child.getHeight());
            }
            subtreeLengths[node.getNr()] = length;
        }
        return subtreeLengths;
    }

    /** Returns the nodes in pre-order, so that every node precedes its children. */
    private Node[] computePreOrder() {
        Node[] preOrder = new Node[this.tree.getNodeCount()];
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(this.tree.getRoot());

        int numVisited = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            preOrder[numVisited++] = node;
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                stack.push(node.getChild(childIndex));
            }
        }
        return preOrder;
    }

    /** Stores the state of every site at the root, which the mutations above the root and the reference sequence determine. */
    private void computeRootStates() {
        System.arraycopy(this.mutations.getReferenceSequence(), 0, this.rootStates, 0, this.rootStates.length);
        for (Mutation mutation : this.mutations.getMutations(this.tree.getRoot())) {
            this.rootStates[mutation.site()] = mutation.newState();
        }
    }

    /** Returns the rate of leaving the given state under the unscaled rate matrix. */
    private double getEscapeRate(int state) {
        return -this.process.getRateMatrix()[state * this.numStates + state];
    }

    /* Category Posteriors */

    /**
     * Computes the posterior of the sites without mutations for every root state. Such a site has
     * no mutations, so its escape time is the one of its root state times the total length.
     */
    private void computeInvariantPosteriors() {
        for (int state = 0; state < this.numStates; state++) {
            double escapeTime = this.getEscapeRate(state) * this.totalLength;
            for (int category = 0; category < this.numCategories; category++) {
                this.invariantLogProbabilities[state][category] = this.computeLogCategoryWeight(escapeTime, 0, this.categoryProportions[category], category);
            }
            normaliseLogProbabilities(this.invariantLogProbabilities[state], this.invariantProbabilities[state]);
        }
    }

    /**
     * Writes the unnormalised log weights of the categories of the given site into the given array,
     * from its statistics and the proportions of the last update.
     */
    private void computeLogProbabilities(int site, double[] logWeights) {
        for (int category = 0; category < this.numCategories; category++) {
            logWeights[category] = this.computeLogCategoryWeight(this.escapeTimes[site], this.numMutations[site], this.categoryProportions[category], category);
        }
    }

    /**
     * Returns the log of the weight w_c r_c^M exp(-r_c L) of a category, given the escape time L
     * and the number of mutations M of a site. Without mutations, the power of the rate is one, so
     * an invariant category keeps its weight.
     */
    private double computeLogCategoryWeight(double escapeTime, int numSiteMutations, double categoryProportion, int category) {
        double categoryRate = this.process.getCategoryRate(category);

        double logWeight = Math.log(categoryProportion) - categoryRate * escapeTime;
        if (numSiteMutations > 0) {
            logWeight += numSiteMutations * Math.log(categoryRate);
        }
        return logWeight;
    }

    /**
     * Normalises the given log weights in place, so that they are log probabilities, and writes
     * the probabilities into the given array if it is not null.
     */
    private static void normaliseLogProbabilities(double[] logWeights, double[] probabilities) {
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        for (double logWeight : logWeights) {
            maxLogWeight = Math.max(maxLogWeight, logWeight);
        }

        double sum = 0.0;
        for (double logWeight : logWeights) {
            sum += Math.exp(logWeight - maxLogWeight);
        }

        double logNormalisation = maxLogWeight + Math.log(sum);
        for (int category = 0; category < logWeights.length; category++) {
            logWeights[category] -= logNormalisation;
            if (probabilities != null) {
                probabilities[category] = Math.exp(logWeights[category]);
            }
        }
    }

}
