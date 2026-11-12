package esat.operators.states;

import beast.base.evolution.alignment.Sequence;
import beast.base.inference.OperatorSchedule;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MissingCladeStateGibbsOperatorTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";

    // E and F are missing together at sites 4 to 9, and B, C, D at some other sites
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTNNNNNNGT"),
            new Sequence("F", "GCGYNNNNNNTT")
    );

    private final Random random = new Random(31);

    /**
     * Runs the operator in the store, accept and restore cycle of MCMC, with and without a
     * bound on the number of sites. The incrementally updated prior must match a computation
     * from scratch after every step, the tips must stay compatible with their data, and a
     * restore must bring back the previous states and prior.
     */
    @Test
    public void testProposalsKeepPriorConsistent() {
        for (int maxSites : new int[]{Integer.MAX_VALUE, 2}) {
            for (int maxLevels : new int[]{Integer.MAX_VALUE, 0, 1}) {
                this.runChain(maxLevels, maxSites);
            }
        }
    }

    private void runChain(int maxLevels, int maxSites) {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        fixture.initialiseStates(fixture.drawCompatibleSequences(this.random));
        NodeStates nodeStates = fixture.nodeStates();
        GeneticPrior geneticPrior = fixture.geneticPrior();

        MissingCladeStateGibbsOperator operator = new MissingCladeStateGibbsOperator();
        operator.initByName("nodeStates", nodeStates, "geneticPrior", geneticPrior, "maxLevels", maxLevels, "maxSites", maxSites, "weight", 1.0);
        operator.setOperatorSchedule(new OperatorSchedule());

        State state = new State();
        state.initByName("stateNode", fixture.tree(), "stateNode", nodeStates);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);
        double logP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        Randomizer.setSeed(11);
        for (long i = 0; i < 1000; i++) {
            int[][] sequencesBefore = fixture.getSequences();
            state.store(i);

            assertEquals(Double.POSITIVE_INFINITY, operator.proposal());

            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();
            double newLogP = geneticPrior.calculateLogP();

            assertTrue(Double.isFinite(newLogP));
            assertTrue(fixture.areTipsCompatible());
            assertEquals(fixture.computeLogPFromScratch(), newLogP, 1e-9);

            if (this.random.nextBoolean()) {
                state.acceptCalculationNodes();
                logP = newLogP;
            } else {
                state.restore();
                state.restoreCalculationNodes();
                assertTrue(Arrays.deepEquals(sequencesBefore, fixture.getSequences()));
                assertEquals(logP, geneticPrior.getCurrentLogP(), 1e-9);
            }

            state.setEverythingDirty(false);
        }
    }

}
