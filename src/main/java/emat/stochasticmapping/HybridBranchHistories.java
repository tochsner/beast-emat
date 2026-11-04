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
 * the rate matrix s Q of the substitution process that a site that starts and ends in state a
 * has mutations. Then the history of every chosen site is drawn from its exact conditional
 * distribution given its end states and, for an agreeing site, that it is not empty. As r does
 * not depend on the states, the agreeing sites are chosen by geometric skipping, so the cost
 * grows with the number of mutations rather than with the genome length. The Jukes-Cantor rate
 * is not used.
 */
public class HybridBranchHistories implements BranchHistorySampler {

    private final SubstitutionProcess process;
    private final int numStates;
    private final int numSites;
    private final UniformisedStochasticMapping siteMapping;

    // reusable transition probabilities exp(s Q t) of the branch of the current call
    private final double[] transitionProbabilities;

    // per site, the scan of the density that last saw it, with the heights and states of its first and last mutation
    private final int[] siteScanStamps;
    private int scanStamp;
    private final double[] siteFirstHeights;
    private final double[] siteLastHeights;
    private final int[] siteStartStates;
    private final int[] siteEndStates;

    // the sites with mutations of the current scan, and how many of them start and end in each pair of states
    private final int[] scannedSites;
    private int numScannedSites;
    private final int[] endStatePairCounts;

    public HybridBranchHistories(SubstitutionProcess process, int numSites) {
        this.process = process;
        this.numStates = process.getNumStates();
        this.numSites = numSites;
        this.siteMapping = new UniformisedStochasticMapping(this.numStates);
        this.transitionProbabilities = new double[this.numStates * this.numStates];

        this.siteScanStamps = new int[numSites];
        this.siteFirstHeights = new double[numSites];
        this.siteLastHeights = new double[numSites];
        this.siteStartStates = new int[numSites];
        this.siteEndStates = new int[numSites];
        this.scannedSites = new int[numSites];
        this.endStatePairCounts = new int[this.numStates * this.numStates];
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
            int startState = differingSites.getStartState(slot);
            int endState = differingSites.getEndState(slot);
            branchMutations.addAll(this.siteMapping.sampleSiteHistory(
                    nodeNr, differingSites.getSite(slot), branchStartHeight, branchEndHeight, startState, endState, rateScale,
                    this.transitionProbabilities[startState * this.numStates + endState]
            ));
        }

        // choose the agreeing sites by skipping over the ones that stay empty

        double selectionProbability = this.computeSelectionProbability(duration, rateScale);
        if (selectionProbability > 0.0) {
            double logSkipProbability = Math.log1p(-selectionProbability);
            double site = 0.0;

            while (true) {
                site += Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / logSkipProbability);
                if (site >= this.numSites) {
                    break;
                }

                int siteIndex = (int) site;
                if (!differingSites.containsSite(siteIndex)) {
                    int state = agreeingStates.applyAsInt(siteIndex);
                    branchMutations.addAll(this.siteMapping.sampleNonEmptySameStateHistory(
                            nodeNr, siteIndex, branchStartHeight, branchEndHeight, state, rateScale,
                            this.computeNonEmptyProbability(state, duration, rateScale)
                    ));
                }
                site++;
            }
        }

        branchMutations.sort(Mutation.BY_DESCENDING_TIME);
        return branchMutations;
    }

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch. Every site with mutations contributes the conditional density of its path, and
     * the agreeing sites contribute the probability of choosing exactly the ones with mutations.
     * <p>
     * The path densities of all sites together only depend on the transition counts and the
     * time spent in each state: a site with start state a spends ∫ s Q_{x(t) x(t)} dt =
     * s (Q_aa t + Σ_m (Q_{b_m b_m} - Q_{a_m a_m}) (t_m - t_end)) in its states, where mutation m
     * at height t_m leads from a_m to b_m. The conditions only depend on the start and end
     * states of every site. Hence, the density takes a single pass over the mutations in any
     * order, and one condition per pair of start and end states.
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
        double diagonalIntegral = 0.0;

        for (int i = 0; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
            int oldState = mutation.oldState();
            int newState = mutation.newState();

            logDensity += Math.log(rateScale * rateMatrix[oldState * this.numStates + newState]);
            diagonalIntegral += (rateMatrix[newState * this.numStates + newState] - rateMatrix[oldState * this.numStates + oldState])
                    * (mutation.time() - branchEndHeight);
            this.recordSiteEnds(mutation);
        }

        // every site with mutations stays in its start state until its first mutation and is conditioned on its end states

        Arrays.fill(this.endStatePairCounts, 0);
        for (int i = 0; i < this.numScannedSites; i++) {
            int site = this.scannedSites[i];
            int startState = this.siteStartStates[site];
            diagonalIntegral += rateMatrix[startState * this.numStates + startState] * duration;
            this.endStatePairCounts[startState * this.numStates + this.siteEndStates[site]]++;
        }
        logDensity += rateScale * diagonalIntegral;

        int numChosenAgreeingSites = 0;
        for (int startState = 0; startState < this.numStates; startState++) {
            for (int endState = 0; endState < this.numStates; endState++) {
                int count = this.endStatePairCounts[startState * this.numStates + endState];
                if (count == 0) {
                    continue;
                }

                double conditionProbability;
                if (startState == endState) {
                    conditionProbability = this.computeNonEmptyProbability(startState, duration, rateScale);
                    numChosenAgreeingSites += count;
                } else {
                    conditionProbability = this.transitionProbabilities[startState * this.numStates + endState];
                }
                logDensity -= count * Math.log(conditionProbability);
            }
        }

        // the agreeing sites are chosen independently with the same probability

        double selectionProbability = this.computeSelectionProbability(duration, rateScale);
        int numAgreeingSites = this.numSites - numDifferingSites;
        if (numChosenAgreeingSites > 0) {
            logDensity += numChosenAgreeingSites * Math.log(selectionProbability);
        }
        logDensity += (numAgreeingSites - numChosenAgreeingSites) * Math.log1p(-selectionProbability);

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

    /** Takes Q from the substitution process and computes the transition probabilities exp(s Q t) of the branch. */
    private void updateModel(double duration, double rateScale) {
        this.siteMapping.setRateMatrix(this.process.getRateMatrix());
        this.process.computeTransitionProbabilities(rateScale * duration, this.transitionProbabilities);
    }

    /**
     * Computes the probability r = Σ_a π_a p_a of choosing an agreeing site, where
     * p_a = (P_aa(t) - exp(s Q_aa t)) / P_aa(t) is the probability that a site that starts and
     * ends in state a has mutations. The frequencies π weight the states, so r depends only on
     * the model and the branch, and both directions of a move use the same r for a branch.
     */
    private double computeSelectionProbability(double duration, double rateScale) {
        double[] frequencies = this.process.getFrequencies();
        double selectionProbability = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            double keepProbability = this.transitionProbabilities[state * this.numStates + state];
            selectionProbability += frequencies[state] * Math.max(this.computeNonEmptyProbability(state, duration, rateScale), 0.0) / keepProbability;
        }
        return Math.min(selectionProbability, 1.0);
    }

    /** Computes the exact probability P_ii(t) - exp(R_ii t) of returning to the given state with at least one mutation. */
    private double computeNonEmptyProbability(int state, double duration, double rateScale) {
        int diagonal = state * this.numStates + state;
        return this.transitionProbabilities[diagonal] - Math.exp(rateScale * this.process.getRateMatrix()[diagonal] * duration);
    }

}
