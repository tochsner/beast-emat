package esat.initialisation;

import beast.base.evolution.alignment.Sequence;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SampledStatesInitialiserTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";

    // the alignment has ambiguous (R, Y) and missing (N, -) sites, which the initial states of the tips must fill in compatibly
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTTNNNNNGT"),
            new Sequence("F", "GCGYTCGAACTT")
    );

    @Test
    public void testInitialisesCompatibleStates() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);

        SampledStatesInitialiser initialiser = new SampledStatesInitialiser();
        initialiser.initByName("nodeStates", fixture.nodeStates(), "geneticPrior", fixture.geneticPrior());

        Randomizer.setSeed(8);
        initialiser.initStateNodes();

        assertTrue(fixture.areTipsCompatible());
        assertTrue(fixture.nodeStates().isEveryBranchDirty());
        double logP = fixture.geneticPrior().calculateLogP();
        assertTrue(Double.isFinite(logP));
        assertEquals(fixture.computeLogPFromScratch(), logP, 1e-9);

        List<StateNode> stateNodes = new ArrayList<>();
        initialiser.getInitialisedStateNodes(stateNodes);
        assertEquals(List.of(fixture.nodeStates()), stateNodes);
    }

}
