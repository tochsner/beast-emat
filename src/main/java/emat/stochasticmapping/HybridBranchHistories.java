package emat.stochasticmapping;

import beast.base.util.Randomizer;
import emat.helper.SiteChanges;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * Samples the history of all sites on a branch given the states at both of its ends in two
 * steps, and computes the probability of proposing a given history. First, the sites that
 * receive mutations are chosen: every site whose end states differ, and every site whose end
 * states agree with the probability r = Σ_a π_a p_a, where p_a is the exact probability under
 * the rate matrix s r_c Q of the substitution process that a site of category c that starts and
 * ends in state a has mutations. Then the history of every chosen site is drawn from its exact
 * conditional distribution given its end states and, for an agreeing site, that it is not empty.
 * The probability r depends on the category of a site, so the agreeing sites are chosen by
 * geometric skipping within every category. As r does not depend on the states, the cost grows
 * with the number of mutations rather than with the genome length. The Jukes-Cantor rate is not
 * used.
 */
public class HybridBranchHistories implements BranchHistorySampler {

    private final SubstitutionProcess process;
    private final int numStates;
    private final int numCategories;
    private final UniformisedStochasticMapping siteMapping;

    // per category: the scale of Q on the branch of the current call, and the transition probabilities exp(scale · t) of the branch
    private final double[] rateScales;
    private final double[][] transitionProbabilities;

    // per site, the scan of the density that last saw it, with the heights and states of its first and last mutation
    private final int[] siteScanStamps;
    private int scanStamp;
    private final double[] siteFirstHeights;
    private final double[] siteLastHeights;
    private final int[] siteStartStates;
    private final int[] siteEndStates;

    // the sites with mutations of the current scan, and how many of them start and end in each pair of states, laid out by category
    private final int[] scannedSites;
    private int numScannedSites;
    private final int[] endStatePairCounts;

    // per category of the current scan: the integral of the escape rates of its sites, and its differing and chosen agreeing sites
    private final double[] diagonalIntegrals;
    private final int[] numDifferingSitesPerCategory;
    private final int[] numChosenAgreeingSitesPerCategory;

    public HybridBranchHistories(SubstitutionProcess process, int numSites) {
        this.process = process;
        this.numStates = process.getNumStates();
        this.numCategories = process.getNumCategories();
        this.siteMapping = new UniformisedStochasticMapping(this.numStates);
        this.rateScales = new double[this.numCategories];
        this.transitionProbabilities = new double[this.numCategories][this.numStates * this.numStates];

        this.siteScanStamps = new int[numSites];
        this.siteFirstHeights = new double[numSites];
        this.siteLastHeights = new double[numSites];
        this.siteStartStates = new int[numSites];
        this.siteEndStates = new int[numSites];
        this.scannedSites = new int[numSites];
        this.endStatePairCounts = new int[this.numCategories * this.numStates * this.numStates];
        this.diagonalIntegrals = new double[this.numCategories];
        this.numDifferingSitesPerCategory = new int[this.numCategories];
        this.numChosenAgreeingSitesPerCategory = new int[this.numCategories];
    }

    /** Samples the history of every site on the branch with the paths of the chosen sites drawn exactly. */
    @Override
    public List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double rateScale,
                                              double jukesCantorRate, SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        List<Mutation> branchMutations = new ArrayList<>();
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return branchMutations;
        }

        this.updateModel(duration, rateScale);

        // the differing sites always receive mutations

        for (int slot = 0; slot < differingSites.getSize(); slot++) {
            int site = differingSites.getSite(slot);
            int category = this.process.getCategory(site);
            int startState = differingSites.getStartState(slot);
            int endState = differingSites.getEndState(slot);
            branchMutations.addAll(this.siteMapping.sampleSiteHistory(
                    nodeNr, site, branchStartHeight, branchEndHeight, startState, endState, this.rateScales[category],
                    this.transitionProbabilities[category][startState * this.numStates + endState]
            ));
        }

        // choose the agreeing sites of every category by skipping over the ones that stay empty

        for (int category = 0; category < this.numCategories; category++) {
            double selectionProbability = this.computeSelectionProbability(duration, category);
            if (selectionProbability <= 0.0) {
                continue;
            }

            int[] sites = this.process.getSitesOfCategory(category);
            double logSkipProbability = Math.log1p(-selectionProbability);
            double position = 0.0;

            while (true) {
                position += Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / logSkipProbability);
                if (position >= sites.length) {
                    break;
                }

                int site = sites[(int) position];
                if (!differingSites.containsSite(site)) {
                    int state = agreeingStates.applyAsInt(site);
                    branchMutations.addAll(this.siteMapping.sampleNonEmptySameStateHistory(
                            nodeNr, site, branchStartHeight, branchEndHeight, state, this.rateScales[category],
                            this.computeNonEmptyProbability(state, duration, category)
                    ));
                }
                position++;
            }
        }

        branchMutations.sort(Mutation.BY_DESCENDING_TIME);
        return branchMutations;
    }

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch. Every site with mutations contributes the conditional density of its path, and
     * the agreeing sites of every category contribute the probability of choosing exactly the
     * ones with mutations.
     * <p>
     * The path densities of all sites of a category together only depend on the transition
     * counts and the time spent in each state: a site with start state a spends ∫ s r_c Q_{x(t) x(t)} dt =
     * s r_c (Q_aa t + Σ_m (Q_{b_m b_m} - Q_{a_m a_m}) (t_m - t_end)) in its states, where mutation m
     * at height t_m leads from a_m to b_m. The conditions only depend on the category and the
     * start and end states of every site. Hence, the density takes a single pass over the
     * mutations in any order, and one condition per category and pair of start and end states.
     */
    @Override
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                 double rateScale, double jukesCantorRate, int numDifferingSites) {
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            return 0.0;
        }

        this.updateModel(duration, rateScale);
        double[] rateMatrix = this.process.getRateMatrix();

        // every mutation contributes its rate, and the change of the escape rate it causes lasts until the branch end

        this.startSiteScan();
        double logDensity = 0.0;
        Arrays.fill(this.diagonalIntegrals, 0.0);

        for (int i = 0; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
            int category = this.process.getCategory(mutation.site());
            int oldState = mutation.oldState();
            int newState = mutation.newState();

            logDensity += Math.log(this.rateScales[category] * rateMatrix[oldState * this.numStates + newState]);
            this.diagonalIntegrals[category] += (rateMatrix[newState * this.numStates + newState] - rateMatrix[oldState * this.numStates + oldState])
                    * (mutation.time() - branchEndHeight);
            this.recordSiteEnds(mutation);
        }

        // every site with mutations stays in its start state until its first mutation and is conditioned on its end states

        int numStatePairs = this.numStates * this.numStates;
        Arrays.fill(this.endStatePairCounts, 0);
        for (int i = 0; i < this.numScannedSites; i++) {
            int site = this.scannedSites[i];
            int category = this.process.getCategory(site);
            int startState = this.siteStartStates[site];
            this.diagonalIntegrals[category] += rateMatrix[startState * this.numStates + startState] * duration;
            this.endStatePairCounts[category * numStatePairs + startState * this.numStates + this.siteEndStates[site]]++;
        }

        Arrays.fill(this.numDifferingSitesPerCategory, 0);
        Arrays.fill(this.numChosenAgreeingSitesPerCategory, 0);
        for (int category = 0; category < this.numCategories; category++) {
            logDensity += this.rateScales[category] * this.diagonalIntegrals[category];

            for (int startState = 0; startState < this.numStates; startState++) {
                for (int endState = 0; endState < this.numStates; endState++) {
                    int count = this.endStatePairCounts[category * numStatePairs + startState * this.numStates + endState];
                    if (count == 0) {
                        continue;
                    }

                    double conditionProbability;
                    if (startState == endState) {
                        conditionProbability = this.computeNonEmptyProbability(startState, duration, category);
                        this.numChosenAgreeingSitesPerCategory[category] += count;
                    } else {
                        conditionProbability = this.transitionProbabilities[category][startState * this.numStates + endState];
                        this.numDifferingSitesPerCategory[category] += count;
                    }
                    logDensity -= count * Math.log(conditionProbability);
                }
            }
        }

        // the agreeing sites of every category are chosen independently with the same probability

        for (int category = 0; category < this.numCategories; category++) {
            double selectionProbability = this.computeSelectionProbability(duration, category);
            int numChosenAgreeingSites = this.numChosenAgreeingSitesPerCategory[category];
            int numAgreeingSites = this.process.getSitesOfCategory(category).length - this.numDifferingSitesPerCategory[category];

            if (numChosenAgreeingSites > 0) {
                logDensity += numChosenAgreeingSites * Math.log(selectionProbability);
            }
            logDensity += (numAgreeingSites - numChosenAgreeingSites) * Math.log1p(-selectionProbability);
        }

        return logDensity;
    }

    /** Starts a new scan of the sites with mutations, which only needs resetting the stamps once the counter runs out. */
    private void startSiteScan() {
        if (this.scanStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.siteScanStamps, 0);
            this.scanStamp = 0;
        }
        this.scanStamp++;
        this.numScannedSites = 0;
    }

    /** Records the given mutation as the first or the last one at its site if it is older or younger than the ones seen so far. */
    private void recordSiteEnds(Mutation mutation) {
        int site = mutation.site();
        double height = mutation.time();

        if (this.siteScanStamps[site] != this.scanStamp) {
            this.siteScanStamps[site] = this.scanStamp;
            this.scannedSites[this.numScannedSites++] = site;
            this.siteFirstHeights[site] = height;
            this.siteLastHeights[site] = height;
            this.siteStartStates[site] = mutation.oldState();
            this.siteEndStates[site] = mutation.newState();
            return;
        }

        if (height > this.siteFirstHeights[site]) {
            this.siteFirstHeights[site] = height;
            this.siteStartStates[site] = mutation.oldState();
        }
        if (height < this.siteLastHeights[site]) {
            this.siteLastHeights[site] = height;
            this.siteEndStates[site] = mutation.newState();
        }
    }

    /**
     * Takes Q from the substitution process and computes, for every category, the scale of Q
     * on the branch and the transition probabilities exp(scale · t) of the branch.
     */
    private void updateModel(double duration, double branchRate) {
        this.siteMapping.setRateMatrix(this.process.getRateMatrix());
        for (int category = 0; category < this.numCategories; category++) {
            this.rateScales[category] = branchRate * this.process.getCategoryRate(category);
            this.process.computeTransitionProbabilities(this.rateScales[category] * duration, this.transitionProbabilities[category]);
        }
    }

    /**
     * Computes the probability r = Σ_a π_a p_a of choosing an agreeing site of the given
     * category, where p_a = (P_aa(t) - exp(s r_c Q_aa t)) / P_aa(t) is the probability that a
     * site that starts and ends in state a has mutations. The frequencies π weight the states,
     * so r depends only on the model, the branch and the category, and both directions of a
     * move use the same r for a branch.
     */
    private double computeSelectionProbability(double duration, int category) {
        double[] frequencies = this.process.getFrequencies();
        double[] categoryProbabilities = this.transitionProbabilities[category];
        double selectionProbability = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            double keepProbability = categoryProbabilities[state * this.numStates + state];
            selectionProbability += frequencies[state] * Math.max(this.computeNonEmptyProbability(state, duration, category), 0.0) / keepProbability;
        }
        return Math.min(selectionProbability, 1.0);
    }

    /** Computes the exact probability P_ii(t) - exp(R_ii t) of returning to the given state with at least one mutation, for the given category. */
    private double computeNonEmptyProbability(int state, double duration, int category) {
        int diagonal = state * this.numStates + state;
        return this.transitionProbabilities[category][diagonal]
                - Math.exp(this.rateScales[category] * this.process.getRateMatrix()[diagonal] * duration);
    }

}
