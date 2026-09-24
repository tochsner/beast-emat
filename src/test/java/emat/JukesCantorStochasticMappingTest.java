package emat;

import beast.base.util.Randomizer;
import emat.helper.JukesCantorStochasticMapping;
import emat.state.Mutation;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class JukesCantorStochasticMappingTest {

    private static final int NUM_STATES = 4;
    private static final int NUM_SITES = 6;

    // the heights of both ends of the branch
    private static final double START_HEIGHT = 1.5;
    private static final double END_HEIGHT = 0.5;

    // the largest number of jumps per site that is compared with its exact probability
    private static final int MAX_COMPARED_JUMPS = 4;

    /**
     * For a short branch, where the agreeing sites are almost always skipped, checks the
     * distribution of the number of jumps per site against its exact conditional distribution.
     */
    @Test
    public void testShortBranchMatchesExactDistribution() {
        this.checkMatchesExactDistribution(0.4);
    }

    /** Same as above for a long branch, where the truncated Poisson draws use rejection. */
    @Test
    public void testLongBranchMatchesExactDistribution() {
        this.checkMatchesExactDistribution(1.7);
    }

    /**
     * For a single site, checks that the density of all histories integrates to one, i.e.
     * that summing the density over the number of jumps, the valid jump chains and the
     * ordered jump times gives one for agreeing as well as differing end states.
     */
    @Test
    public void testDensityIsNormalised() {
        JukesCantorStochasticMapping mapping = new JukesCantorStochasticMapping(NUM_STATES, 1);
        double rate = 0.9;
        double duration = START_HEIGHT - END_HEIGHT;

        for (int numDifferingSites = 0; numDifferingSites <= 1; numDifferingSites++) {
            double total = 0.0;
            double timeVolume = 1.0;

            for (int numJumps = 0; numJumps < 60; numJumps++) {
                if (numJumps > 0) {
                    timeVolume *= duration / numJumps;
                }

                double numChains = numDifferingSites == 0 ? this.getNumReturningChains(numJumps) : this.getNumChangingChains(numJumps);
                double logDensity = mapping.computeLogBranchHistoryDensity(
                        this.getDummyMutations(numJumps), START_HEIGHT, END_HEIGHT, rate, numDifferingSites
                );

                total += Math.exp(logDensity) * numChains * timeVolume;
            }

            assertEquals(1.0, total, 1e-10);
        }
    }

    private void checkMatchesExactDistribution(double rate) {
        Randomizer.setSeed(1);

        JukesCantorStochasticMapping mapping = new JukesCantorStochasticMapping(NUM_STATES, NUM_SITES);
        Map<Integer, int[]> differingSites = Map.of(1, new int[]{0, 2}, 4, new int[]{3, 1});

        // sample histories and count the jumps per site, pooling the agreeing and the differing sites

        int numSamples = 200000;
        double[] agreeingCounts = new double[MAX_COMPARED_JUMPS + 1];
        double[] differingCounts = new double[MAX_COMPARED_JUMPS + 1];

        for (int i = 0; i < numSamples; i++) {
            List<Mutation> branchMutations = mapping.sampleBranchHistory(
                    7, START_HEIGHT, END_HEIGHT, rate, differingSites, site -> site % NUM_STATES
            );
            this.checkHistory(branchMutations, differingSites);

            int[] numJumps = new int[NUM_SITES];
            for (Mutation mutation : branchMutations) {
                numJumps[mutation.site()]++;
            }

            for (int site = 0; site < NUM_SITES; site++) {
                if (numJumps[site] > MAX_COMPARED_JUMPS) {
                    continue;
                }
                if (differingSites.containsKey(site)) {
                    differingCounts[numJumps[site]] += 1.0 / (differingSites.size() * numSamples);
                } else {
                    agreeingCounts[numJumps[site]] += 1.0 / ((NUM_SITES - differingSites.size()) * numSamples);
                }
            }
        }

        // compare with P(k | a, b) = Pois(k; μ̃ t) (number of chains from a to b) / (K - 1)^k / P_ab(t)

        double expectedJumps = rate * (START_HEIGHT - END_HEIGHT);
        double stayProbability = (1.0 + (NUM_STATES - 1) * Math.exp(-expectedJumps * NUM_STATES / (NUM_STATES - 1.0))) / NUM_STATES;
        double changeProbability = (1.0 - stayProbability) / (NUM_STATES - 1);

        double poissonProbability = Math.exp(-expectedJumps);
        for (int k = 0; k <= MAX_COMPARED_JUMPS; k++) {
            if (k > 0) {
                poissonProbability *= expectedJumps / k;
            }
            double chainProbability = poissonProbability / Math.pow(NUM_STATES - 1, k);

            assertEquals(chainProbability * this.getNumReturningChains(k) / stayProbability, agreeingCounts[k], 0.005);
            assertEquals(chainProbability * this.getNumChangingChains(k) / changeProbability, differingCounts[k], 0.005);
        }
    }

    /**
     * Checks that the mutations are sorted by descending height within the branch, and that
     * the mutations of every site form a chain from its start to its end state.
     */
    private void checkHistory(List<Mutation> branchMutations, Map<Integer, int[]> differingSites) {
        int[] states = new int[NUM_SITES];
        double[] previousHeights = new double[NUM_SITES];
        for (int site = 0; site < NUM_SITES; site++) {
            states[site] = differingSites.containsKey(site) ? differingSites.get(site)[0] : site % NUM_STATES;
            previousHeights[site] = START_HEIGHT;
        }

        double previousHeight = START_HEIGHT;
        for (Mutation mutation : branchMutations) {
            assertTrue(mutation.time() <= previousHeight && mutation.time() >= END_HEIGHT);
            assertEquals(states[mutation.site()], mutation.oldState());
            assertEquals(previousHeights[mutation.site()], mutation.timeOfPreviousMutation());

            states[mutation.site()] = mutation.newState();
            previousHeights[mutation.site()] = mutation.time();
            previousHeight = mutation.time();
        }

        for (int site = 0; site < NUM_SITES; site++) {
            int endState = differingSites.containsKey(site) ? differingSites.get(site)[1] : site % NUM_STATES;
            assertEquals(endState, states[site]);
        }
    }

    /* Helpers */

    /** Returns the number of jump chains with k jumps that return to their start state. */
    private double getNumReturningChains(int k) {
        return (Math.pow(NUM_STATES - 1, k) + (NUM_STATES - 1) * Math.pow(-1, k)) / NUM_STATES;
    }

    /** Returns the number of jump chains with k jumps that end in a given other state. */
    private double getNumChangingChains(int k) {
        return (Math.pow(NUM_STATES - 1, k) - Math.pow(-1, k)) / NUM_STATES;
    }

    private List<Mutation> getDummyMutations(int numJumps) {
        List<Mutation> mutations = new ArrayList<>();
        for (int i = 0; i < numJumps; i++) {
            mutations.add(new Mutation(0, END_HEIGHT, START_HEIGHT, 0, 0, 1));
        }
        return mutations;
    }

}
