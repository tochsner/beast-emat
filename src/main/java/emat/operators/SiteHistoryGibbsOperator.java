package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import emat.helper.SiteHistorySampler;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the complete mutational history of a random site from its exact conditional " +
        "distribution (stochastic mapping). Sites are picked proportional to their Fitch parsimony score on " +
        "the UPGMA tree of the alignment plus a pseudo count, as an estimate of their expected number of " +
        "mutations. This is a Gibbs move, so it is always accepted.")
public class SiteHistoryGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> pseudoCountInput = new Input<>("pseudoCount", "the weight added to the parsimony score of every site, so that sites without parsimony changes are also resampled", 1.0);

    Mutations mutations;
    TreeInterface tree;
    SiteHistorySampler siteHistorySampler;

    // the cumulative selection weights of the sites
    double[] cumulativeSiteWeights;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.siteHistorySampler = new SiteHistorySampler(this.mutations, this.geneticPriorInput.get());
        this.cumulativeSiteWeights = this.computeCumulativeSiteWeights();
    }

    /**
     * Picks a site proportional to its weight and replaces its history by a draw from the
     * conditional distribution of the genetic prior given the tip data and everything else.
     * The weights do not depend on the state, so as a Gibbs move it is always accepted.
     */
    @Override
    public double proposal() {
        int site = this.sampleSite();

        this.siteHistorySampler.updateModel();
        List<List<Mutation>> newSiteMutations = this.siteHistorySampler.sampleSiteHistory(site);

        for (Node node : this.tree.getNodesAsArray()) {
            this.replaceSiteMutations(node, site, newSiteMutations.get(node.getNr()));
        }

        return Double.POSITIVE_INFINITY;
    }

    /* Site Selection */

    /**
     * Computes the cumulative selection weights of the sites. The weight of a site is its
     * Fitch parsimony score on the UPGMA tree of the alignment, the minimum number of
     * mutations needed to explain the tip data, plus the pseudo count.
     */
    private double[] computeCumulativeSiteWeights() {
        double pseudoCount = this.pseudoCountInput.get();
        if (pseudoCount < 0.0) {
            throw new IllegalArgumentException("The pseudo count must not be negative.");
        }

        int[] siteScores = FitchParsimony.computeOnUpgmaTree(this.mutations.getAlignment()).getSiteScores();

        double[] cumulativeWeights = new double[siteScores.length];
        double totalWeight = 0.0;
        for (int site = 0; site < siteScores.length; site++) {
            totalWeight += siteScores[site] + pseudoCount;
            cumulativeWeights[site] = totalWeight;
        }

        if (totalWeight <= 0.0) {
            throw new IllegalArgumentException("All sites have zero weight. Use a positive pseudo count.");
        }

        return cumulativeWeights;
    }

    /** Samples a site proportional to its weight. */
    private int sampleSite() {
        double totalWeight = this.cumulativeSiteWeights[this.cumulativeSiteWeights.length - 1];
        double threshold = Randomizer.nextDouble() * totalWeight;

        // find the first site whose cumulative weight exceeds the threshold, which skips sites with zero weight

        int low = 0;
        int high = this.cumulativeSiteWeights.length - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (this.cumulativeSiteWeights[middle] > threshold) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }

        return low;
    }

    /* Site Histories */

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

}
