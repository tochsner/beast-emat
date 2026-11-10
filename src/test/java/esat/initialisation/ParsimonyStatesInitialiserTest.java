package esat.initialisation;

import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.inference.StateNode;
import emat.helper.FitchParsimony;
import esat.EsatFixture;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ParsimonyStatesInitialiserTest {

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

    /** The reconstruction is compatible with the tips and has as many changes as the Fitch parsimony score of the tree. */
    @Test
    public void testInitialisesParsimoniousStates() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);

        ParsimonyStatesInitialiser initialiser = new ParsimonyStatesInitialiser();
        initialiser.initByName("nodeStates", fixture.nodeStates());
        initialiser.initStateNodes();

        assertTrue(fixture.areTipsCompatible());
        double logP = fixture.geneticPrior().calculateLogP();
        assertTrue(Double.isFinite(logP));
        assertEquals(fixture.computeLogPFromScratch(), logP, 1e-9);

        int parsimonyScore = 0;
        for (int score : new FitchParsimony(fixture.alignment(), fixture.tree()).getSiteScores()) {
            parsimonyScore += score;
        }
        assertEquals(parsimonyScore, fixture.nodeStates().getDifferenceCount());

        List<StateNode> stateNodes = new ArrayList<>();
        initialiser.getInitialisedStateNodes(stateNodes);
        assertEquals(List.of(fixture.nodeStates()), stateNodes);
    }

    /** The clock rate is set to the number of changes per site and unit of tree length. */
    @Test
    public void testInitialisesClockRate() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);

        ParsimonyStatesInitialiser initialiser = new ParsimonyStatesInitialiser();
        initialiser.initByName("nodeStates", fixture.nodeStates(), "clockRate", fixture.clockRate());
        initialiser.initStateNodes();

        double treeLength = 0.0;
        for (Node node : fixture.tree().getNodesAsArray()) {
            if (!node.isRoot()) {
                treeLength += node.getLength();
            }
        }
        double expectedRate = fixture.nodeStates().getDifferenceCount() / (fixture.alignment().getSiteCount() * treeLength);
        assertEquals(expectedRate, fixture.clockRate().get(), 1e-12);

        List<StateNode> stateNodes = new ArrayList<>();
        initialiser.getInitialisedStateNodes(stateNodes);
        assertEquals(List.of(fixture.nodeStates(), fixture.clockRate()), stateNodes);
    }

}
