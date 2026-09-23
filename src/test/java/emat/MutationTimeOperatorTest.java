package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MutationTimeOperatorTest {

    /**
     * Applies the operator many times to a branch with two mutations at one site and one
     * mutation at another site. Every proposal resamples one mutation uniformly given its
     * neighbours, so the chain samples the times uniformly under the ordering constraints:
     * the two mutations at the same site follow the order statistics of two uniform draws
     * on the branch, and the single mutation is uniform on the branch.
     */
    @Test
    public void testSamplesUniformOrderStatistics() {
        Randomizer.setSeed(1);

        // B is the first taxon, so its sequence is the reference and the root sequence

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("B", "AC"),
                        new Sequence("A", "TA")
                ),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "(A:2,B:2);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        // A's branch runs from height 2 to 0 and carries A -> G -> T at site 0 and C -> A at site 1

        Node nodeA = this.getTip(tree, "A");
        int nodeNr = nodeA.getNr();

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            mutationsAboveNode.add(new ArrayList<>());
        }
        mutationsAboveNode.get(nodeNr).add(new Mutation(nodeNr, 1.5, 2.0, 0, 0, 2));
        mutationsAboveNode.get(nodeNr).add(new Mutation(nodeNr, 1.0, 2.0, 1, 1, 0));
        mutationsAboveNode.get(nodeNr).add(new Mutation(nodeNr, 0.5, 1.5, 0, 2, 3));

        mutations.initialiseMutations(mutationsAboveNode);

        MutationTimeOperator operator = new MutationTimeOperator();
        operator.initByName("weight", 1.0, "mutations", mutations);

        // run the operator and average the times

        int numBurnin = 1000;
        int numSamples = 200000;

        double sumFirstTime = 0.0;
        double sumSecondTime = 0.0;
        double sumOtherSiteTime = 0.0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            operator.proposal();

            if (i < numBurnin) {
                continue;
            }

            for (Mutation mutation : mutations.getMutations(nodeA)) {
                if (mutation.site() == 1) {
                    sumOtherSiteTime += mutation.time();
                } else if (mutation.oldState() == 0) {
                    sumFirstTime += mutation.time();
                } else {
                    sumSecondTime += mutation.time();
                }
            }
        }

        // the order statistics of two uniform draws on [0, 2] have means 4/3 and 2/3

        assertEquals(4.0 / 3.0, sumFirstTime / numSamples, 0.02);
        assertEquals(2.0 / 3.0, sumSecondTime / numSamples, 0.02);
        assertEquals(1.0, sumOtherSiteTime / numSamples, 0.02);
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
