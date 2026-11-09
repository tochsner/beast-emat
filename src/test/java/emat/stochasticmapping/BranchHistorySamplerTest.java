package emat.stochasticmapping;

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
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks that the branch history samplers, which draw the jump times of all sites on a branch
 * at once, propose the same histories as a reference that maps every site on its own with the
 * single-site samplers and sorts the mutations of the branch. Both are compared by summaries of
 * many proposals on one branch: the number of mutations per site, the heights of the first
 * mutations per site and on the branch, and the order in which the sites first mutate.
 */
public class BranchHistorySamplerTest {

    private static final int NUM_STATES = 4;
    private static final int NUM_SAMPLES = 200000;
    private static final int NUM_SITES = 20;
    private static final double BRANCH_START_HEIGHT = 2.0;
    private static final double BRANCH_END_HEIGHT = 0.5;
    private static final double RATE_SCALE = 0.7;

    // sites 0 to 2 differ between the ends of the branch, and site 3 is an agreeing site
    private static final int[][] DIFFERING_SITES = {{0, 0, 1}, {1, 0, 2}, {2, 1, 3}};
    private static final int AGREEING_SITE = 3;

    // the summaries of a history, whose means are compared
    private static final int NUM_STATISTICS = 12;

    /**
     * Compares the Jukes-Cantor branch sampler with the Jukes-Cantor mapping of every single
     * site under a rate matrix whose average escape rate is one, so that both use the rate s.
     */
    @Test
    public void testJukesCantorMatchesSingleSites() {
        JukesCantorStochasticMapping siteMapping = new JukesCantorStochasticMapping(NUM_STATES, NUM_SITES);
        siteMapping.setRateMatrix(this.createJukesCantorRateMatrix());

        this.assertSamplerMatchesReference(new JukesCantorStochasticMapping(NUM_STATES, NUM_SITES), () -> {
            List<Mutation> branchMutations = new ArrayList<>();
            for (int site = 0; site < NUM_SITES; site++) {
                branchMutations.addAll(siteMapping.sampleSiteHistory(0, site, BRANCH_START_HEIGHT, BRANCH_END_HEIGHT,
                        this.getStartState(site), this.getEndState(site), RATE_SCALE, Double.NaN));
            }
            return this.sortByDescendingHeight(branchMutations);
        });
    }

    /**
     * Compares the hybrid branch sampler under HKY with unequal frequencies, whose states have
     * different escape rates, with exact single-site mappings: every differing site is mapped
     * exactly, and every agreeing site is chosen with the probability r of the hybrid sampler
     * and then mapped exactly, rejecting empty histories.
     */
    @Test
    public void testHybridMatchesSingleSites() {
        SubstitutionProcess process = this.createHkyProcess();
        UniformisedStochasticMapping siteMapping = new UniformisedStochasticMapping(NUM_STATES);
        siteMapping.setRateMatrix(process.getRateMatrix());

        double duration = BRANCH_START_HEIGHT - BRANCH_END_HEIGHT;
        double[] transitionProbabilities = new double[NUM_STATES * NUM_STATES];
        process.computeTransitionProbabilities(RATE_SCALE * duration, transitionProbabilities);
        double selectionProbability = this.computeSelectionProbability(process, transitionProbabilities, duration);

        this.assertSamplerMatchesReference(new HybridBranchHistories(process, NUM_SITES), () -> {
            List<Mutation> branchMutations = new ArrayList<>();
            for (int site = 0; site < NUM_SITES; site++) {
                int startState = this.getStartState(site);
                int endState = this.getEndState(site);
                double endProbability = transitionProbabilities[startState * NUM_STATES + endState];

                if (startState != endState) {
                    branchMutations.addAll(siteMapping.sampleSiteHistory(0, site, BRANCH_START_HEIGHT, BRANCH_END_HEIGHT,
                            startState, endState, RATE_SCALE, endProbability));
                } else if (Randomizer.nextDouble() < selectionProbability) {
                    List<Mutation> siteMutations;
                    do {
                        siteMutations = siteMapping.sampleSiteHistory(0, site, BRANCH_START_HEIGHT, BRANCH_END_HEIGHT,
                                startState, endState, RATE_SCALE, endProbability);
                    } while (siteMutations.isEmpty());
                    branchMutations.addAll(siteMutations);
                }
            }
            return this.sortByDescendingHeight(branchMutations);
        });
    }

    private void assertSamplerMatchesReference(BranchHistorySampler sampler, Supplier<List<Mutation>> referenceSampler) {
        Randomizer.setSeed(7);

        SiteChanges differingSites = new SiteChanges(NUM_SITES);
        for (int[] differingSite : DIFFERING_SITES) {
            differingSites.addSite(differingSite[0], differingSite[1], differingSite[2]);
        }

        double[][] sampled = this.sampleStatistics(() -> sampler.sampleBranchHistory(
                0, BRANCH_START_HEIGHT, BRANCH_END_HEIGHT, RATE_SCALE, RATE_SCALE, differingSites, site -> site % NUM_STATES));
        double[][] reference = this.sampleStatistics(referenceSampler);

        for (int statistic = 0; statistic < NUM_STATISTICS; statistic++) {
            double standardError = Math.sqrt((reference[1][statistic] + sampled[1][statistic]) / NUM_SAMPLES);
            assertEquals(reference[0][statistic], sampled[0][statistic], 4.5 * standardError + 1e-12,
                    "mean of statistic " + statistic);
        }
    }

    /**
     * Draws many histories on the branch, checks that each is valid, and returns the mean and
     * the variance of every statistic.
     */
    private double[][] sampleStatistics(Supplier<List<Mutation>> historySampler) {
        double[] sums = new double[NUM_STATISTICS];
        double[] squareSums = new double[NUM_STATISTICS];

        for (int sample = 0; sample < NUM_SAMPLES; sample++) {
            List<Mutation> branchMutations = historySampler.get();
            this.assertValidHistory(branchMutations);

            double[] statistics = this.computeStatistics(branchMutations);
            for (int statistic = 0; statistic < NUM_STATISTICS; statistic++) {
                sums[statistic] += statistics[statistic];
                squareSums[statistic] += statistics[statistic] * statistics[statistic];
            }
        }

        double[][] moments = new double[2][NUM_STATISTICS];
        for (int statistic = 0; statistic < NUM_STATISTICS; statistic++) {
            moments[0][statistic] = sums[statistic] / NUM_SAMPLES;
            moments[1][statistic] = squareSums[statistic] / NUM_SAMPLES - moments[0][statistic] * moments[0][statistic];
        }
        return moments;
    }

    /**
     * Checks that the mutations are sorted by descending height within the branch, and that
     * the mutations of every site form a path between its end states with consistent previous
     * heights.
     */
    private void assertValidHistory(List<Mutation> branchMutations) {
        int[] states = new int[NUM_SITES];
        double[] previousHeights = new double[NUM_SITES];
        for (int site = 0; site < NUM_SITES; site++) {
            states[site] = this.getStartState(site);
            previousHeights[site] = Mutation.BRANCH_START;
        }

        double previousHeight = BRANCH_START_HEIGHT;
        for (Mutation mutation : branchMutations) {
            assertTrue(mutation.time() <= previousHeight && mutation.time() >= BRANCH_END_HEIGHT);
            assertEquals(states[mutation.site()], mutation.oldState());
            assertTrue(mutation.oldState() != mutation.newState());
            assertEquals(previousHeights[mutation.site()], mutation.timeOfPreviousMutation());

            states[mutation.site()] = mutation.newState();
            previousHeights[mutation.site()] = mutation.time();
            previousHeight = mutation.time();
        }

        for (int site = 0; site < NUM_SITES; site++) {
            assertEquals(this.getEndState(site), states[site]);
        }
    }

    /**
     * Computes the statistics of a history: the total number of mutations, the number at sites
     * 0, 1 and the agreeing site, the height of the first mutation at sites 0, 1 and on the
     * branch, the height of the last mutation on the branch, whether the branch first mutates
     * at site 0 or at site 1, whether site 0 mutates before site 1, and the number of agreeing
     * sites with mutations.
     */
    private double[] computeStatistics(List<Mutation> branchMutations) {
        double[] statistics = new double[NUM_STATISTICS];
        double[] firstHeights = new double[NUM_SITES];
        boolean[] hasMutations = new boolean[NUM_SITES];

        for (Mutation mutation : branchMutations) {
            int site = mutation.site();
            if (!hasMutations[site]) {
                hasMutations[site] = true;
                firstHeights[site] = mutation.time();
            }
        }

        statistics[0] = branchMutations.size();
        statistics[1] = this.countMutations(branchMutations, 0);
        statistics[2] = this.countMutations(branchMutations, 1);
        statistics[3] = this.countMutations(branchMutations, AGREEING_SITE);
        statistics[4] = firstHeights[0];
        statistics[5] = firstHeights[1];
        statistics[6] = branchMutations.getFirst().time();
        statistics[7] = branchMutations.getLast().time();
        statistics[8] = branchMutations.getFirst().site() == 0 ? 1.0 : 0.0;
        statistics[9] = branchMutations.getFirst().site() == 1 ? 1.0 : 0.0;
        statistics[10] = firstHeights[0] > firstHeights[1] ? 1.0 : 0.0;

        for (int site = 0; site < NUM_SITES; site++) {
            if (hasMutations[site] && this.getStartState(site) == this.getEndState(site)) {
                statistics[11]++;
            }
        }
        return statistics;
    }

    private int countMutations(List<Mutation> branchMutations, int site) {
        int count = 0;
        for (Mutation mutation : branchMutations) {
            if (mutation.site() == site) {
                count++;
            }
        }
        return count;
    }

    /* Branch Setup */

    private int getStartState(int site) {
        for (int[] differingSite : DIFFERING_SITES) {
            if (differingSite[0] == site) {
                return differingSite[1];
            }
        }
        return site % NUM_STATES;
    }

    private int getEndState(int site) {
        for (int[] differingSite : DIFFERING_SITES) {
            if (differingSite[0] == site) {
                return differingSite[2];
            }
        }
        return site % NUM_STATES;
    }

    private List<Mutation> sortByDescendingHeight(List<Mutation> branchMutations) {
        branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return branchMutations;
    }

    /**
     * Computes the probability r = Σ_a π_a p_a with which the hybrid sampler chooses an agreeing
     * site, where p_a = (P_aa(t) - exp(s Q_aa t)) / P_aa(t).
     */
    private double computeSelectionProbability(SubstitutionProcess process, double[] transitionProbabilities, double duration) {
        double selectionProbability = 0.0;
        for (int state = 0; state < NUM_STATES; state++) {
            int diagonal = state * NUM_STATES + state;
            double keepProbability = transitionProbabilities[diagonal];
            double emptyProbability = Math.exp(RATE_SCALE * process.getRateMatrix()[diagonal] * duration);
            selectionProbability += process.getFrequencies()[state] * (keepProbability - emptyProbability) / keepProbability;
        }
        return selectionProbability;
    }

    /** Creates the Jukes-Cantor rate matrix with an escape rate of one from every state. */
    private double[] createJukesCantorRateMatrix() {
        double[] rateMatrix = new double[NUM_STATES * NUM_STATES];
        for (int from = 0; from < NUM_STATES; from++) {
            for (int to = 0; to < NUM_STATES; to++) {
                rateMatrix[from * NUM_STATES + to] = from == to ? -1.0 : 1.0 / (NUM_STATES - 1);
            }
        }
        return rateMatrix;
    }

    /** Creates the substitution process of HKY with unequal frequencies, whose states have different escape rates. */
    private SubstitutionProcess createHkyProcess() {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "ACGT".repeat(NUM_SITES / 4)), new Sequence("B", "ACGT".repeat(NUM_SITES / 4))
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(A:1.0,B:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

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

}
