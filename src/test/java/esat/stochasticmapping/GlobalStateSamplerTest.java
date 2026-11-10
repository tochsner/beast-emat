package esat.stochasticmapping;

import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import org.junit.jupiter.api.Test;

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
                    int state = sample.changesAtNode()[nodeNr].getStateOrDefault(site, sample.rootSequence()[site]);
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
        double chiSquare = 0.0;
        int numCells = 0;
        double pooledObserved = 0.0;
        double pooledExpected = 0.0;
        for (int cell = 0; cell < probabilities.length; cell++) {
            double expected = NUM_SAMPLES * probabilities[cell];
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
