package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.util.Randomizer;
import emat.operators.IdenticalTipPermutationOperator;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IdenticalTipPermutationOperatorTest {

    /**
     * Applies the operator many times to a tree with the identical tips A, B and C and a
     * different tip X. The positions are below P1 (height 3), below P2 (height 2), and below
     * P2 with two mutations down to height 1.2. B has height 1.5, so it can only take the
     * first two positions, while A and C at height 0 can take any. This leaves four valid
     * permutations, which the operator must visit uniformly.
     */
    @Test
    public void testSamplesValidPermutationsUniformly() {
        Randomizer.setSeed(1);

        // A is the first taxon, so its sequence is the reference and the root sequence

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "AC"),
                        new Sequence("B", "AC"),
                        new Sequence("C", "AC"),
                        new Sequence("X", "TC")
                ),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "((A:3,X:2):1,(B:0.5,C:2):2);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        // X carries A -> T at site 0, and C carries C -> G -> C at site 1

        Node nodeX = this.getTip(tree, "X");
        Node nodeC = this.getTip(tree, "C");

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            mutationsAboveNode.add(new ArrayList<>());
        }
        mutationsAboveNode.get(nodeX.getNr()).add(new Mutation(nodeX.getNr(), 2.0, 3.0, 0, 0, 3));
        mutationsAboveNode.get(nodeC.getNr()).add(new Mutation(nodeC.getNr(), 1.8, 2.0, 1, 1, 2));
        mutationsAboveNode.get(nodeC.getNr()).add(new Mutation(nodeC.getNr(), 1.2, 1.8, 1, 2, 1));

        mutations.initialiseMutations(mutationsAboveNode);

        IdenticalTipPermutationOperator operator = new IdenticalTipPermutationOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations);

        Node parentOne = nodeX.getParent();

        // run the operator and count the configurations, given by the parent of B and the tip with the mutations

        int numSamples = 200000;
        Map<String, Integer> counts = new HashMap<>();

        for (int i = 0; i < numSamples; i++) {
            operator.proposal();

            Node nodeB = this.getTip(tree, "B");
            assertTrue(nodeB.getParent().getHeight() > nodeB.getHeight());
            assertSame(parentOne, nodeX.getParent());

            String mutatedTip = null;
            for (String taxon : List.of("A", "B", "C")) {
                Node tip = this.getTip(tree, taxon);
                if (!mutations.getMutations(tip).isEmpty()) {
                    mutatedTip = taxon;
                    assertEquals(2, mutations.getMutations(tip).size());
                    assertEquals(tip.getNr(), mutations.getMutations(tip).get(0).nodeNr());
                }
            }

            String configuration = (nodeB.getParent() == parentOne ? "P1" : "P2") + "-" + mutatedTip;
            counts.merge(configuration, 1, Integer::sum);
        }

        // B below P1 leaves A and C for the two positions below P2, and B below P2 leaves A and C for P1 and the mutated position

        assertEquals(4, counts.size());
        for (String configuration : List.of("P1-A", "P1-C", "P2-A", "P2-C")) {
            assertEquals(0.25, counts.getOrDefault(configuration, 0) / (double) numSamples, 0.01);
        }
    }

    /** Returns the tip with the given taxon name. */
    private Node getTip(TreeParser tree, String taxonName) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxonName)) {
                return node;
            }
        }
        throw new IllegalArgumentException("No tip named " + taxonName + ".");
    }

}
