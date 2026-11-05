package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.spec.inference.parameter.IntVectorParam;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.SubstitutionProcess;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

@Description("Resamples the rate category of every site from its exact conditional distribution given the tree " +
        "and the mutations. Given the histories, the genetic prior factorises over sites, and the category c of a site " +
        "with weight w_c in the site model has the probability proportional to w_c r_c^M exp(-r_c L), where r_c is the " +
        "rate of the category, M the number of mutations of the site on the non-root branches, and L the rate-weighted " +
        "time that the site spends in its escape rates. As the categories are drawn exactly, the move is a Gibbs move " +
        "that is always accepted. Requires a genetic prior with siteRateCategories.")
public class SiteRateCategoryGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model and holds the categories", Input.Validate.REQUIRED);

    Mutations mutations;
    TreeInterface tree;
    GeneticPrior geneticPrior;
    IntVectorParam<?> siteRateCategories;
    SubstitutionProcess process;

    int numSites;
    int numCategories;

    // per site: the number of mutations on the non-root branches, and the rate-weighted time of its escape rates
    int[] numMutations;
    double[] escapeTimes;

    // reusable weights of the categories of a site
    double[] categoryWeights;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.geneticPrior = this.geneticPriorInput.get();
        this.siteRateCategories = this.geneticPrior.getSiteRateCategories();
        if (this.siteRateCategories == null) {
            throw new IllegalArgumentException("The genetic prior has no siteRateCategories to resample.");
        }

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        this.process = new SubstitutionProcess(this.geneticPrior, this.tree, numStates);

        this.numSites = this.mutations.getReferenceSequence().length;
        this.numCategories = this.geneticPrior.getNumCategories();
        this.numMutations = new int[this.numSites];
        this.escapeTimes = new double[this.numSites];
        this.categoryWeights = new double[this.numCategories];
    }

    /**
     * Collects the statistics of every site, and then draws the category of every site from its
     * conditional distribution, which only changes the categories that are drawn differently.
     */
    @Override
    public double proposal() {
        this.process.update();
        this.collectSiteStatistics();

        double[] categoryProportions = this.geneticPrior.siteModel.getCategoryProportions(this.tree.getRoot());
        for (int site = 0; site < this.numSites; site++) {
            int category = this.sampleCategory(site, categoryProportions);
            if (category != this.siteRateCategories.get(site)) {
                this.siteRateCategories.set(site, category);
            }
        }

        return Double.POSITIVE_INFINITY;
    }

    /* Site Statistics */

    /**
     * Computes for every site the number of its mutations on the non-root branches and the
     * rate-weighted time L of its escape rates. The escape rate of a state is the one of the
     * unscaled rate matrix. With the states x_s(t) of a site, L is the integral over the tree of
     * the escape rate of x_s(t), weighted by the branch rates. It starts at the root state
     * times the total weighted length, and every non-root mutation changes the escape rate for
     * the part of the tree below it, which is the weighted length of its branch below the
     * mutation plus the weighted length below the branch.
     */
    private void collectSiteStatistics() {
        Node root = this.tree.getRoot();
        double[] subtreeLengths = this.computeSubtreeLengths();
        int[] rootStates = this.computeRootStates();

        Arrays.fill(this.numMutations, 0);
        double totalLength = subtreeLengths[root.getNr()];
        for (int site = 0; site < this.numSites; site++) {
            this.escapeTimes[site] = this.getEscapeRate(rootStates[site]) * totalLength;
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

    /** Returns the state of every site at the root, which the mutations above the root and the reference sequence determine. */
    private int[] computeRootStates() {
        int[] rootStates = this.mutations.getReferenceSequence().clone();
        for (Mutation mutation : this.mutations.getMutations(this.tree.getRoot())) {
            rootStates[mutation.site()] = mutation.newState();
        }
        return rootStates;
    }

    /** Returns the rate of leaving the given state under the unscaled rate matrix. */
    private double getEscapeRate(int state) {
        int numStates = this.process.getNumStates();
        return -this.process.getRateMatrix()[state * numStates + state];
    }

    /* Category Sampling */

    /** Draws the category of the given site with probability proportional to its weight. */
    private int sampleCategory(int site, double[] categoryProportions) {
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        for (int category = 0; category < this.numCategories; category++) {
            this.categoryWeights[category] = this.computeLogCategoryWeight(site, category, categoryProportions[category]);
            maxLogWeight = Math.max(maxLogWeight, this.categoryWeights[category]);
        }

        // subtract the maximum before exponentiating, which keeps the weights of all categories from underflowing
        for (int category = 0; category < this.numCategories; category++) {
            this.categoryWeights[category] = Math.exp(this.categoryWeights[category] - maxLogWeight);
        }
        return StochasticMapping.sampleIndex(this.categoryWeights);
    }

    /**
     * Returns the log of the weight w_c r_c^M exp(-r_c L) of the given category for the given site.
     * Without mutations, the power of the rate is one, so an invariant category keeps its weight.
     */
    private double computeLogCategoryWeight(int site, int category, double categoryProportion) {
        double categoryRate = this.process.getCategoryRate(category);

        double logWeight = Math.log(categoryProportion) - categoryRate * this.escapeTimes[site];
        if (this.numMutations[site] > 0) {
            logWeight += this.numMutations[site] * Math.log(categoryRate);
        }
        return logWeight;
    }

}
