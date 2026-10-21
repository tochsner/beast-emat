package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import emat.stochasticmapping.JukesCantorStochasticMapping;
import emat.helper.SiteHistorySampler;
import emat.helper.SiteMutations;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.UniformisedStochasticMapping;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Resamples the complete mutational history of a random site by stochastic mapping. Sites are " +
        "picked by the site selection strategy: uniformly, or proportional to a score of their variability plus " +
        "a pseudo count. The score is the Fitch parsimony score on the UPGMA tree of the alignment, as an " +
        "estimate of the expected number of mutations. The missing strategy adds a weight for the fraction of " +
        "tips whose data is missing or ambiguous at the site, as their states are resampled with the history. With the exact sampler, the history is " +
        "drawn from its exact conditional distribution, so the move is a Gibbs move that is always accepted. " +
        "With the Jukes-Cantor sampler, the node states are drawn exactly, but the paths on the branches are " +
        "drawn under Jukes-Cantor (docs/mcmc-moves.md §3.4), and the move is accepted by Metropolis-Hastings.")
public class SiteHistoryGibbsOperator extends Operator {

    /** The stochastic mapping used for the paths on the branches. */
    public enum Sampler {
        EXACT,
        JUKES_CANTOR
    }

    /** The strategy for choosing the site. */
    public enum SiteSelection {
        UNIFORM,
        PARSIMONY,
        MISSING
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> pseudoCountInput = new Input<>("pseudoCount", "the weight added to the score of every site, so that sites with a score of zero are also resampled. Not used by UNIFORM", 1.0);
    final public Input<Double> missingWeightInput = new Input<>("missingWeight", "the weight added to a site per unit of the fraction of tips with missing or ambiguous data at the site. Only used by MISSING", 20.0);
    final public Input<Integer> maxNumSamplesInput = new Input<>("maxNumSamples", "the maximum number of sequences used to build the UPGMA tree for the parsimony scores. Larger alignments are uniformly subsampled", 1000);
    final public Input<SiteSelection> siteSelectionInput = new Input<>("siteSelection", "the strategy for choosing the site: UNIFORM picks every site with the same probability, PARSIMONY proportional to its Fitch parsimony score on the UPGMA tree of the alignment plus the pseudo count, MISSING additionally adds the missing weight times the fraction of tips with missing or ambiguous data at the site", SiteSelection.PARSIMONY, SiteSelection.values());
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
            SiteMutations.replace(this.mutations, node, site, newSiteMutations.get(node.getNr()), this);
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
     * Computes the cumulative selection weights of the sites. With uniform selection, every
     * site has the same weight. Otherwise, the weight of a site is its score plus the pseudo
     * count, where the score is its Fitch parsimony score on the UPGMA tree of the
     * alignment, the minimum number of mutations needed to explain the tip data. Alignments
     * with more than the maximum number of samples are subsampled to build the tree.
     */
    private double[] computeCumulativeSiteWeights() {
        double pseudoCount = this.pseudoCountInput.get();
        if (pseudoCount < 0.0) {
            throw new IllegalArgumentException("The pseudo count must not be negative.");
        }

        int numSites = this.mutations.getReferenceSequence().length;
        double[] siteWeights = new double[numSites];

        switch (this.siteSelectionInput.get()) {
            case UNIFORM -> Arrays.fill(siteWeights, 1.0);
            case PARSIMONY -> {
                int[] siteScores = FitchParsimony.computeOnUpgmaTree(this.mutations.getAlignment(), this.maxNumSamplesInput.get()).getSiteScores();
                for (int site = 0; site < numSites; site++) {
                    siteWeights[site] = siteScores[site] + pseudoCount;
                }
            }
            case MISSING -> {
                int[] siteScores = FitchParsimony.computeOnUpgmaTree(this.mutations.getAlignment(), this.maxNumSamplesInput.get()).getSiteScores();
                double[] unresolvedFractions = this.computeUnresolvedFractions();
                double missingWeight = this.missingWeightInput.get();
                if (missingWeight < 0.0) {
                    throw new IllegalArgumentException("The missing weight must not be negative.");
                }
                for (int site = 0; site < numSites; site++) {
                    siteWeights[site] = siteScores[site] + pseudoCount + missingWeight * unresolvedFractions[site];
                }
            }
        }

        double[] cumulativeWeights = new double[numSites];
        double totalWeight = 0.0;
        for (int site = 0; site < numSites; site++) {
            totalWeight += siteWeights[site];
            cumulativeWeights[site] = totalWeight;
        }

        if (totalWeight <= 0.0) {
            throw new IllegalArgumentException("All sites have zero weight. Use a positive pseudo count.");
        }

        return cumulativeWeights;
    }

    /**
     * Computes for every site the fraction of tips whose data is missing or ambiguous, i.e.
     * compatible with more than one state. This only depends on the alignment, so the
     * selection of the site stays independent of the state.
     */
    private double[] computeUnresolvedFractions() {
        Alignment alignment = this.mutations.getAlignment();
        DataType dataType = alignment.getDataType();
        int numTaxa = alignment.getTaxonCount();

        double[] patternFractions = new double[alignment.getPatternCount()];
        for (int patternNr = 0; patternNr < patternFractions.length; patternNr++) {
            int numUnresolved = 0;
            for (int taxonNr = 0; taxonNr < numTaxa; taxonNr++) {
                if (dataType.getStatesForCode(alignment.getPattern(taxonNr, patternNr)).length > 1) {
                    numUnresolved++;
                }
            }
            patternFractions[patternNr] = (double) numUnresolved / numTaxa;
        }

        double[] siteFractions = new double[alignment.getSiteCount()];
        for (int site = 0; site < siteFractions.length; site++) {
            siteFractions[site] = patternFractions[alignment.getPatternIndex(site)];
        }
        return siteFractions;
    }

    /** Returns the probability that the given site is picked by a proposal. */
    public double getSiteProbability(int site) {
        double previousWeight = site == 0 ? 0.0 : this.cumulativeSiteWeights[site - 1];
        return (this.cumulativeSiteWeights[site] - previousWeight) / this.cumulativeSiteWeights[this.cumulativeSiteWeights.length - 1];
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
            siteMutations.add(SiteMutations.collect(this.mutations, this.tree.getNode(nodeNr), site));
        }
        return siteMutations;
    }

}
