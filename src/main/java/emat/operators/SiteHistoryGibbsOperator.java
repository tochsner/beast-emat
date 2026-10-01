package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import emat.stochasticmapping.JukesCantorStochasticMapping;
import emat.helper.SiteHistorySampler;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.UniformisedStochasticMapping;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the complete mutational history of a random site by stochastic mapping. Sites are " +
        "picked proportional to their Fitch parsimony score on the UPGMA tree of the alignment plus a pseudo " +
        "count, as an estimate of their expected number of mutations. With the exact sampler, the history is " +
        "drawn from its exact conditional distribution, so the move is a Gibbs move that is always accepted. " +
        "With the Jukes-Cantor sampler, the node states are drawn exactly, but the paths on the branches are " +
        "drawn under Jukes-Cantor (docs/mcmc-moves.md §3.4), and the move is accepted by Metropolis-Hastings.")
public class SiteHistoryGibbsOperator extends Operator {

    /** The stochastic mapping used for the paths on the branches. */
    public enum Sampler {
        EXACT,
        JUKES_CANTOR
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> pseudoCountInput = new Input<>("pseudoCount", "the weight added to the parsimony score of every site, so that sites without parsimony changes are also resampled", 1.0);
    final public Input<Integer> maxNumSamplesInput = new Input<>("maxNumSamples", "the maximum number of sequences used to build the UPGMA tree for the parsimony scores. Larger alignments are uniformly subsampled", 1000);
    final public Input<Sampler> samplerInput = new Input<>("sampler", "the stochastic mapping used for the paths on the branches: EXACT samples under the model of the genetic prior, JUKES_CANTOR under an approximating Jukes-Cantor model", Sampler.EXACT, Sampler.values());

    Mutations mutations;
    TreeInterface tree;
    Sampler sampler;
    SiteHistorySampler siteHistorySampler;

    // the cumulative selection weights of the sites
    double[] cumulativeSiteWeights;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.sampler = this.samplerInput.get();
        this.siteHistorySampler = new SiteHistorySampler(this.mutations, this.geneticPriorInput.get(), this.createStochasticMapping());
        this.cumulativeSiteWeights = this.computeCumulativeSiteWeights();
    }

    /**
     * Picks a site proportional to its weight and replaces its history by a new draw. With
     * the exact sampler, this is a draw from the conditional distribution of the genetic
     * prior given the tip data and everything else, so as a Gibbs move it is always
     * accepted. Otherwise, the Hastings ratio is the ratio of the probabilities of proposing
     * the old and the new history. The weights do not depend on the state, so the choice of
     * the site is symmetric.
     */
    @Override
    public double proposal() {
        int site = this.sampleSite();

        this.siteHistorySampler.updateModel();
        List<List<Mutation>> newSiteMutations = this.siteHistorySampler.sampleSiteHistory(site);

        double logHastingsRatio = Double.POSITIVE_INFINITY;
        if (this.sampler != Sampler.EXACT) {
            List<List<Mutation>> oldSiteMutations = this.collectSiteMutations(site);
            logHastingsRatio = this.siteHistorySampler.computeLogSiteHistoryDensity(site, oldSiteMutations)
                    - this.siteHistorySampler.computeLogSiteHistoryDensity(site, newSiteMutations);
        }

        for (Node node : this.tree.getNodesAsArray()) {
            this.replaceSiteMutations(node, site, newSiteMutations.get(node.getNr()));
        }

        return logHastingsRatio;
    }

    /** Creates the stochastic mapping of the chosen sampler. */
    private StochasticMapping createStochasticMapping() {
        int numStates = this.mutations.getAlignment().getMaxStateCount();
        int numSites = this.mutations.getReferenceSequence().length;

        return switch (this.sampler) {
            case EXACT -> new UniformisedStochasticMapping(numStates);
            case JUKES_CANTOR -> new JukesCantorStochasticMapping(numStates, numSites);
        };
    }

    /* Site Selection */

    /**
     * Computes the cumulative selection weights of the sites. The weight of a site is its
     * Fitch parsimony score on the UPGMA tree of the alignment, the minimum number of
     * mutations needed to explain the tip data, plus the pseudo count. Alignments with more
     * than the maximum number of samples are subsampled to build the tree.
     */
    private double[] computeCumulativeSiteWeights() {
        double pseudoCount = this.pseudoCountInput.get();
        if (pseudoCount < 0.0) {
            throw new IllegalArgumentException("The pseudo count must not be negative.");
        }

        int[] siteScores = FitchParsimony.computeOnUpgmaTree(this.mutations.getAlignment(), this.maxNumSamplesInput.get()).getSiteScores();

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
     * Collects the current history of the given site, indexed by the number of the node
     * below each branch, in the same form as a sampled history.
     */
    private List<List<Mutation>> collectSiteMutations(int site) {
        List<List<Mutation>> siteMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            List<Mutation> branchSiteMutations = new ArrayList<>();
            for (Mutation mutation : this.mutations.getMutations(this.tree.getNode(nodeNr))) {
                if (mutation.site() == site) {
                    branchSiteMutations.add(mutation);
                }
            }
            siteMutations.add(branchSiteMutations);
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

}
