package esat.stochasticmapping;

import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import esat.state.BranchChanges;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static esat.EsatFixture.NUM_STATES;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GlobalStateSamplerTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";
    private static final int NUM_SAMPLES = 20000;

    // the alignment has ambiguous (R, Y) and missing (N, -) sites
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTTNNNNNGT"),
            new Sequence("F", "GCGYTCGAACTT")
    );

    // four tips with an ambiguous and a missing tip, where the first two sites share a pattern
    private static final String SMALL_NEWICK = "((A:0.3,B:0.5):0.4,(C:0.6,D:0.2):0.5);";
    private static final List<Sequence> SMALL_SEQUENCES = List.of(
            new Sequence("A", "AAC"),
            new Sequence("B", "RRG"),
            new Sequence("C", "NNT"),
            new Sequence("D", "CCA")
    );

    /**
     * Compares the frequencies of the joint configurations of all nodes at every site with the
     * exact posterior from enumerating them. A block size of one puts every pattern into a
     * block of its own.
     */
    @Test
    public void testSamplesMatchExactPosterior() {
        EsatFixture fixture = EsatFixture.create(SMALL_NEWICK, SMALL_SEQUENCES);
        assertEquals(2, fixture.alignment().getPatternCount());
        int numSites = fixture.nodeStates().getNumSites();
        int numNodes = fixture.tree().getNodeCount();
        int numConfigurations = 1 << (2 * numNodes);

        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler sampler = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess, 1);

        long[][] counts = new long[numSites][numConfigurations];
        Randomizer.setSeed(3);
        for (int i = 0; i < NUM_SAMPLES; i++) {
            GlobalStateSampler.Sample sample = sampler.sample();
            assertTrue(sample.changesAtNode()[fixture.tree().getRoot().getNr()].isEmpty());

            for (int site = 0; site < numSites; site++) {
                int configuration = 0;
                for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                    int state = getSampledState(sample.changesAtNode(), fixture.tree().getNode(nodeNr), site, sample.rootSequence()[site]);
                    configuration |= state << (2 * nodeNr);
                }
                counts[site][configuration]++;
            }
        }

        for (int site = 0; site < numSites; site++) {
            double[] posterior = this.computeExactPosterior(fixture, site);
            this.assertCountsMatch(posterior, counts[site], "site " + site);
        }
    }

    /**
     * Samples every pattern on its own and compares the frequencies of the joint configurations
     * of all nodes at every site of it with the exact posterior.
     */
    @Test
    public void testPatternSamplesMatchExactPosterior() {
        EsatFixture fixture = EsatFixture.create(SMALL_NEWICK, SMALL_SEQUENCES);
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler sampler = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess);
        int numConfigurations = 1 << (2 * fixture.tree().getNodeCount());

        Randomizer.setSeed(5);
        for (int patternNr = 0; patternNr < sampler.getPatternCount(); patternNr++) {
            long[][] counts = null;
            int[] sites = null;
            for (int i = 0; i < NUM_SAMPLES; i++) {
                GlobalStateSampler.PatternSample sample = sampler.samplePattern(patternNr);
                if (counts == null) {
                    sites = sample.sites();
                    counts = new long[sites.length][numConfigurations];
                }
                for (int k = 0; k < sites.length; k++) {
                    counts[k][this.getConfiguration(fixture.tree(), sample, k)]++;
                }
            }

            for (int k = 0; k < sites.length; k++) {
                assertEquals(patternNr, fixture.alignment().getPatternIndex(sites[k]));
                this.assertCountsMatch(this.computeExactPosterior(fixture, sites[k]), counts[k], "site " + sites[k]);
            }
        }
    }

    /**
     * Samples a pattern of many sites on long branches, so that every node deviates from its
     * default state at many sites. The configurations pooled over the sites must follow the
     * exact posterior, and the number of sites at which a node deviates from its most probable
     * state must have the mean and variance of a binomial distribution, as the sites are
     * independent.
     */
    @Test
    public void testRepeatedPatternMatchesExactPosterior() {
        int numSites = 400;
        int numSamples = 2000;
        EsatFixture fixture = EsatFixture.create("((A:1.5,B:2.0):1.0,(C:1.8,D:1.2):1.5);", List.of(
                new Sequence("A", "A".repeat(numSites)),
                new Sequence("B", "R".repeat(numSites)),
                new Sequence("C", "N".repeat(numSites)),
                new Sequence("D", "C".repeat(numSites))
        ));
        assertEquals(1, fixture.alignment().getPatternCount());
        int numNodes = fixture.tree().getNodeCount();

        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler sampler = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess);

        // the most probable state of every node and the probability of any other state

        double[] posterior = this.computeExactPosterior(fixture, 0);
        double[][] marginals = new double[numNodes][NUM_STATES];
        for (int configuration = 0; configuration < posterior.length; configuration++) {
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                marginals[nodeNr][(configuration >> (2 * nodeNr)) & 3] += posterior[configuration];
            }
        }
        int[] modalStates = new int[numNodes];
        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            for (int state = 1; state < NUM_STATES; state++) {
                if (marginals[nodeNr][state] > marginals[nodeNr][modalStates[nodeNr]]) {
                    modalStates[nodeNr] = state;
                }
            }
        }

        long[] counts = new long[posterior.length];
        double[] sumDeviations = new double[numNodes];
        double[] sumSquaredDeviations = new double[numNodes];
        Randomizer.setSeed(6);
        for (int i = 0; i < numSamples; i++) {
            GlobalStateSampler.PatternSample sample = sampler.samplePattern(0);
            int[] deviations = new int[numNodes];
            for (int k = 0; k < numSites; k++) {
                int configuration = this.getConfiguration(fixture.tree(), sample, k);
                counts[configuration]++;
                for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                    if (((configuration >> (2 * nodeNr)) & 3) != modalStates[nodeNr]) {
                        deviations[nodeNr]++;
                    }
                }
            }
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                sumDeviations[nodeNr] += deviations[nodeNr];
                sumSquaredDeviations[nodeNr] += (double) deviations[nodeNr] * deviations[nodeNr];
            }
        }

        this.assertCountsMatch(posterior, counts, "the pooled sites");

        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            double deviationProbability = 1.0 - marginals[nodeNr][modalStates[nodeNr]];
            double expectedMean = numSites * deviationProbability;
            double expectedVariance = numSites * deviationProbability * (1.0 - deviationProbability);

            double mean = sumDeviations[nodeNr] / numSamples;
            double variance = sumSquaredDeviations[nodeNr] / numSamples - mean * mean;

            // an observed tip never deviates
            if (deviationProbability < 1e-12) {
                assertEquals(0.0, mean, "deviations of the observed node " + nodeNr);
                continue;
            }

            // the standard errors of the mean and of the variance of a nearly normal count
            double meanError = Math.sqrt(expectedVariance / numSamples);
            double varianceError = expectedVariance * Math.sqrt(2.0 / numSamples);
            assertEquals(expectedMean, mean, 5.0 * meanError, "mean deviations of node " + nodeNr);
            assertEquals(expectedVariance, variance, 5.0 * varianceError, "variance of deviations of node " + nodeNr);
        }
    }

    /** Returns the configuration of all nodes at the site with the given index of a pattern sample, with two bits per node. */
    private int getConfiguration(Tree tree, GlobalStateSampler.PatternSample sample, int index) {
        int configuration = 0;
        for (int nodeNr = 0; nodeNr < sample.changesAtNode().length; nodeNr++) {
            int state = getSampledState(sample.changesAtNode(), tree.getNode(nodeNr), sample.sites()[index], sample.rootStates()[index]);
            configuration |= state << (2 * nodeNr);
        }
        return configuration;
    }

    /** Returns the state of the given node at the given site, which is the state of the nearest change at or above it, or the given root state. */
    private static int getSampledState(BranchChanges[] changesAtNode, Node node, int site, int rootState) {
        for (Node current = node; !current.isRoot(); current = current.getParent()) {
            int index = changesAtNode[current.getNr()].indexOfSite(site);
            if (index >= 0) {
                return changesAtNode[current.getNr()].getTo(index);
            }
        }
        return rootState;
    }

    /**
     * Computes the posterior of every configuration of all nodes at the given site, indexed by
     * the states of the nodes in two bits each, from the root frequencies and the transition
     * probabilities of the substitution model. Configurations with incompatible tips have
     * probability zero.
     */
    private double[] computeExactPosterior(EsatFixture fixture, int site) {
        int numNodes = fixture.tree().getNodeCount();
        double[] frequencies = fixture.hky().getFrequencies();
        double rate = fixture.clockRate().get();

        double[][] matrices = new double[numNodes][NUM_STATES * NUM_STATES];
        for (Node node : fixture.tree().getNodesAsArray()) {
            if (!node.isRoot()) {
                fixture.hky().getTransitionProbabilities(node, node.getParent().getHeight(), node.getHeight(), rate, matrices[node.getNr()]);
            }
        }

        double[] posterior = new double[1 << (2 * numNodes)];
        double total = 0.0;
        for (int configuration = 0; configuration < posterior.length; configuration++) {
            double probability = 1.0;
            for (Node node : fixture.tree().getNodesAsArray()) {
                int state = (configuration >> (2 * node.getNr())) & 3;
                int compatibleStates = fixture.geneticPrior().getTipCompatibility().getCompatibleStates(node, site);
                if ((compatibleStates & (1 << state)) == 0) {
                    probability = 0.0;
                    break;
                }

                if (node.isRoot()) {
                    probability *= frequencies[state];
                } else {
                    int parentState = (configuration >> (2 * node.getParent().getNr())) & 3;
                    probability *= matrices[node.getNr()][parentState * NUM_STATES + state];
                }
            }
            posterior[configuration] = probability;
            total += probability;
        }

        for (int configuration = 0; configuration < posterior.length; configuration++) {
            posterior[configuration] /= total;
        }
        return posterior;
    }

    /**
     * Compares counts with their expectation by a chi-square statistic. Cells with a small
     * expectation are pooled into one, and cells without probability must stay empty.
     */
    private void assertCountsMatch(double[] probabilities, long[] counts, String description) {
        double numDraws = Arrays.stream(counts).sum();
        double chiSquare = 0.0;
        int numCells = 0;
        double pooledObserved = 0.0;
        double pooledExpected = 0.0;
        for (int cell = 0; cell < probabilities.length; cell++) {
            double expected = numDraws * probabilities[cell];
            double observed = counts[cell];
            if (expected == 0.0) {
                assertEquals(0.0, observed, "Impossible configuration " + cell + " at " + description);
            } else if (expected < 5.0) {
                pooledObserved += observed;
                pooledExpected += expected;
            } else {
                chiSquare += (observed - expected) * (observed - expected) / expected;
                numCells++;
            }
        }
        if (pooledExpected > 0.0) {
            chiSquare += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
            numCells++;
        }

        // the number of cells bounds the degrees of freedom, and the margin lies far in the tail
        double bound = numCells + 6.0 * Math.sqrt(2.0 * numCells) + 10.0;
        assertTrue(chiSquare < bound, "Chi-square " + chiSquare + " above " + bound + " at " + description);
    }

    /** Initialises the states from samples, which must be compatible with the tips and have a finite prior. */
    @Test
    public void testSamplesAreValidStates() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler sampler = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess, 5);

        Randomizer.setSeed(4);
        for (int i = 0; i < 50; i++) {
            GlobalStateSampler.Sample sample = sampler.sample();
            fixture.nodeStates().initialiseStates(sample.rootSequence(), List.of(sample.changesAtNode()));

            assertTrue(fixture.areTipsCompatible());
            double logP = fixture.computeLogPFromScratch();
            assertTrue(Double.isFinite(logP));
        }
    }

    /** Two different observed tips below a branch of length zero are impossible, which the sampler reports. */
    @Test
    public void testImpossibleDataThrows() {
        EsatFixture fixture = EsatFixture.create(
                "((A:0.0,B:0.0):0.5,C:0.5);",
                List.of(new Sequence("A", "A"), new Sequence("B", "C"), new Sequence("C", "G"))
        );
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler sampler = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess);

        assertThrows(IllegalStateException.class, sampler::sample);
    }

}
