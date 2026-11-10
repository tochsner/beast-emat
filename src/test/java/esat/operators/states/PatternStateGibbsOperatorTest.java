package esat.operators.states;

import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.inference.OperatorSchedule;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.NodeStateSampler;
import esat.stochasticmapping.StateEdit;
import esat.stochasticmapping.SubstitutionProcess;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PatternStateGibbsOperatorTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";

    // the alignment has ambiguous (R, Y) and missing (N, -) sites
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTTNNNNNGT"),
            new Sequence("F", "GCGYTCGAACTT")
    );

    private final Random random = new Random(29);

    /**
     * Mixes the operator with single-node Gibbs updates in the store, accept and restore cycle
     * of MCMC. The incrementally updated prior must match a computation from scratch after
     * every step, and a restore must bring back the previous states and prior.
     */
    @Test
    public void testProposalsKeepPriorConsistent() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        fixture.initialiseStates(fixture.drawCompatibleSequences(this.random));
        NodeStates nodeStates = fixture.nodeStates();
        GeneticPrior geneticPrior = fixture.geneticPrior();

        PatternStateGibbsOperator operator = new PatternStateGibbsOperator();
        operator.initByName("nodeStates", nodeStates, "geneticPrior", geneticPrior, "weight", 1.0);
        operator.setOperatorSchedule(new OperatorSchedule());

        SubstitutionProcess substitutionProcess = new SubstitutionProcess(geneticPrior);
        NodeStateSampler nodeStateSampler = new NodeStateSampler(nodeStates, substitutionProcess);

        State state = new State();
        state.initByName("stateNode", fixture.tree(), "stateNode", nodeStates);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);
        double logP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        Randomizer.setSeed(9);
        for (long i = 0; i < 2000; i++) {
            int[][] sequencesBefore = fixture.getSequences();
            state.store(i);

            if (this.random.nextBoolean()) {
                assertEquals(Double.POSITIVE_INFINITY, operator.proposal());
            } else {
                Node node = fixture.tree().getNode(this.random.nextInt(fixture.tree().getNodeCount()));
                substitutionProcess.update();
                StateEdit edit = nodeStateSampler.sample(node);
                nodeStates.setStates(node, edit.sites(), edit.states(), EsatFixture.NO_OPERATOR);
            }

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
