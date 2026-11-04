package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.helper.SiteChanges;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.HybridBranchHistories;
import emat.stochasticmapping.SubstitutionProcess;
import emat.stochasticmapping.UniformisedStochasticMapping;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HybridBranchHistoriesTest {

    private static final int NUM_STATES = 4;
    private static final int NUM_SITES = 60;
    private static final int NUM_HISTORIES = 300;

    private static final double START_HEIGHT = 2.0;
    private static final double END_HEIGHT = 0.5;

    /**
     * Checks the density of sampled histories against a reference that groups the mutations
     * by site and multiplies the conditional densities of the single sites, for branches
     * that are long enough to carry several mutations per site and mutations at agreeing
     * sites. The histories are passed in a random order, on which the density must not depend.
     */
    @Test
    public void testDensityMatchesSiteWiseReference() {
        Randomizer.setSeed(7);

        SubstitutionProcess process = this.createProcess();
        HybridBranchHistories sampler = new HybridBranchHistories(process, NUM_SITES);

        int numCheckedAgreeingSites = 0;
        for (int i = 0; i < NUM_HISTORIES; i++) {
            double rateScale = 0.2 + 1.5 * Randomizer.nextDouble();
            int[] agreeingStates = this.createRandomSequence();
            SiteChanges differingSites = this.createRandomDifferingSites(agreeingStates);

            List<Mutation> history = new ArrayList<>(sampler.sampleBranchHistory(
                    0, START_HEIGHT, END_HEIGHT, rateScale, 0.0, differingSites, site -> agreeingStates[site]
            ));
            numCheckedAgreeingSites += this.countMutatedAgreeingSites(history, differingSites);
            Collections.shuffle(history, new java.util.Random(i));

            double expected = this.computeReferenceLogDensity(process, history, rateScale, differingSites.getSize());
            double actual = sampler.computeLogBranchHistoryDensity(history, START_HEIGHT, END_HEIGHT, rateScale, 0.0, differingSites.getSize());
            assertEquals(expected, actual, 1e-9 * Math.max(1.0, Math.abs(expected)));
        }

        // the comparison has to cover agreeing sites with mutations, whose condition differs
        assertTrue(numCheckedAgreeingSites > 0);
    }

    /**
     * Checks that reusing the terms of the number of jumps for repeated end states and
     * branches draws the same histories as computing them for every site, including after
     * the rate matrix changes.
     */
    @Test
    public void testCachedJumpTermsDrawSameHistories() {
        SubstitutionProcess process = this.createProcess();
        double[] rateMatrix = process.getRateMatrix();
        double[] scaledRateMatrix = rateMatrix.clone();
        for (int i = 0; i < scaledRateMatrix.length; i++) {
            scaledRateMatrix[i] *= 1.7;
        }

        UniformisedStochasticMapping cachedMapping = new UniformisedStochasticMapping(NUM_STATES);
        cachedMapping.setRateMatrix(rateMatrix);

        double[] durations = {0.3, 1.2, 0.3, 2.5};
        for (int call = 0; call < 400; call++) {
            double[] currentRateMatrix = call < 200 ? rateMatrix : scaledRateMatrix;
            cachedMapping.setRateMatrix(currentRateMatrix);

            int startState = call % NUM_STATES;
            int endState = (call / NUM_STATES) % NUM_STATES;
            double duration = durations[(call / 16) % durations.length];

            UniformisedStochasticMapping freshMapping = new UniformisedStochasticMapping(NUM_STATES);
            freshMapping.setRateMatrix(currentRateMatrix);

            Randomizer.setSeed(call + 1);
            List<Mutation> cachedHistory = cachedMapping.sampleSiteHistory(0, call % NUM_SITES, duration, 0.0, startState, endState, 0.9, Double.NaN);
            Randomizer.setSeed(call + 1);
            List<Mutation> freshHistory = freshMapping.sampleSiteHistory(0, call % NUM_SITES, duration, 0.0, startState, endState, 0.9, Double.NaN);

            assertEquals(freshHistory, cachedHistory);
        }
    }

    /* Reference */

    /**
     * Computes the density of the history as the product over the sites with mutations of
     * their conditional path densities, times the probability of choosing exactly the
     * agreeing sites with mutations.
     */
    private double computeReferenceLogDensity(SubstitutionProcess process, List<Mutation> history, double rateScale, int numDifferingSites) {
        double duration = START_HEIGHT - END_HEIGHT;
        double[] rateMatrix = process.getRateMatrix();
        double[] transitionProbabilities = new double[NUM_STATES * NUM_STATES];
        process.computeTransitionProbabilities(rateScale * duration, transitionProbabilities);

        UniformisedStochasticMapping siteMapping = new UniformisedStochasticMapping(NUM_STATES);
        siteMapping.setRateMatrix(rateMatrix);

        List<Mutation> sortedMutations = new ArrayList<>(history);
        sortedMutations.sort(Comparator.comparingInt(Mutation::site).thenComparing(Comparator.comparingDouble(Mutation::time).reversed()));

        double logDensity = 0.0;
        int numChosenAgreeingSites = 0;

        int start = 0;
        while (start < sortedMutations.size()) {
            int site = sortedMutations.get(start).site();
            int end = start;
            while (end < sortedMutations.size() && sortedMutations.get(end).site() == site) {
                end++;
            }

            List<Mutation> siteMutations = sortedMutations.subList(start, end);
            int startState = siteMutations.getFirst().oldState();
            int endState = siteMutations.getLast().newState();

            double conditionProbability;
            if (startState == endState) {
                conditionProbability = this.computeNonEmptyProbability(rateMatrix, transitionProbabilities, startState, duration, rateScale);
                numChosenAgreeingSites++;
            } else {
                conditionProbability = transitionProbabilities[startState * NUM_STATES + endState];
            }

            logDensity += siteMapping.computeLogSiteHistoryDensity(
                    siteMutations, START_HEIGHT, END_HEIGHT, startState, endState, rateScale, conditionProbability
            );
            start = end;
        }

        double[] frequencies = process.getFrequencies();
        double selectionProbability = 0.0;
        for (int state = 0; state < NUM_STATES; state++) {
            selectionProbability += frequencies[state]
                    * Math.max(this.computeNonEmptyProbability(rateMatrix, transitionProbabilities, state, duration, rateScale), 0.0)
                    / transitionProbabilities[state * NUM_STATES + state];
        }
        selectionProbability = Math.min(selectionProbability, 1.0);

        if (numChosenAgreeingSites > 0) {
            logDensity += numChosenAgreeingSites * Math.log(selectionProbability);
        }
        logDensity += (NUM_SITES - numDifferingSites - numChosenAgreeingSites) * Math.log1p(-selectionProbability);

        return logDensity;
    }

    private double computeNonEmptyProbability(double[] rateMatrix, double[] transitionProbabilities, int state, double duration, double rateScale) {
        int diagonal = state * NUM_STATES + state;
        return transitionProbabilities[diagonal] - Math.exp(rateScale * rateMatrix[diagonal] * duration);
    }

    /* Fixture */

    /** Creates the substitution process of HKY with unequal frequencies, whose escape rates differ between the states. */
    private SubstitutionProcess createProcess() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(new Sequence("A", "A".repeat(NUM_SITES)), new Sequence("B", "C".repeat(NUM_SITES))),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(A:1,B:1);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", hky);

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        SubstitutionProcess process = new SubstitutionProcess(geneticPrior, tree, NUM_STATES);
        process.update();
        return process;
    }

    private int[] createRandomSequence() {
        int[] sequence = new int[NUM_SITES];
        for (int site = 0; site < NUM_SITES; site++) {
            sequence[site] = Randomizer.nextInt(NUM_STATES);
        }
        return sequence;
    }

    /** Lets about a third of the sites differ, starting in the given state and ending in another one. */
    private SiteChanges createRandomDifferingSites(int[] startStates) {
        SiteChanges differingSites = new SiteChanges(NUM_SITES);
        for (int site = 0; site < NUM_SITES; site++) {
            if (Randomizer.nextDouble() < 0.3) {
                int endState = (startStates[site] + 1 + Randomizer.nextInt(NUM_STATES - 1)) % NUM_STATES;
                differingSites.addSite(site, startStates[site], endState);
            }
        }
        return differingSites;
    }

    private int countMutatedAgreeingSites(List<Mutation> history, SiteChanges differingSites) {
        return (int) history.stream().mapToInt(Mutation::site).distinct().filter(site -> !differingSites.containsSite(site)).count();
    }

}
