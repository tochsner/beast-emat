package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ExcessMutationLoggerTest {

    // nucleotide states
    private static final int A = 0;
    private static final int C = 1;
    private static final int G = 2;

    private static final String NEWICK = "((A:1.0,B:1.0)X:1.0,C:2.0)R;";

    /**
     * On the tree ((A,B)X,C)R, site 0 is A at all tips and site 1 is G only at C. Site 1 needs
     * a single mutation, so only the A -> C -> A detour on the branch above A at site 0 is
     * in conflict excess.
     */
    @Test
    public void testLogsOnlyMutationsBeyondParsimony() {
        TreeParser tree = this.createTree();
        Mutations mutations = this.createMutations(tree, new String[] {"AA", "AA", "AG"}, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "A"), 0.6, Mutation.BRANCH_START, 0, A, C),
                new Mutation(this.getNodeNr(tree, "A"), 0.3, 0.6, 0, C, A),
                new Mutation(this.getNodeNr(tree, "C"), 0.5, Mutation.BRANCH_START, 1, A, G)
        });

        int nodeNr = this.getNodeNr(tree, "A");
        assertEquals("0:2@" + nodeNr + "(1,0.0)," + nodeNr + "(1,0.0)\t-", this.logExcess(mutations));
    }

    @Test
    public void testLogsPlaceholderWithoutExcess() {
        TreeParser tree = this.createTree();
        Mutations mutations = this.createMutations(tree, new String[] {"AA", "AA", "AG"}, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "C"), 0.5, Mutation.BRANCH_START, 1, A, G)
        });

        assertEquals("-\t-", this.logExcess(mutations));
    }

    /**
     * Tip B is missing at site 0, so its mutation there only counts as missing excess. The
     * mutation on the branch above A at site 1 is still in conflict with the data of A.
     */
    @Test
    public void testSeparatesMutationsOnMissingTips() {
        TreeParser tree = this.createTree();
        Mutations mutations = this.createMutations(tree, new String[] {"AA", "NA", "AA"}, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "B"), 0.5, Mutation.BRANCH_START, 0, A, C),
                new Mutation(this.getNodeNr(tree, "A"), 0.6, Mutation.BRANCH_START, 1, A, C),
                new Mutation(this.getNodeNr(tree, "A"), 0.3, 0.6, 1, C, A)
        });

        int aNr = this.getNodeNr(tree, "A");
        int bNr = this.getNodeNr(tree, "B");
        assertEquals("1:2@" + aNr + "(1,0.0)," + aNr + "(1,0.0)\t0:1@" + bNr + "(1,0.0)", this.logExcess(mutations));
    }

    /** Both tips below X are missing at site 0, so the mutation above X needs no parsimony change. */
    @Test
    public void testTreatsCladesOfMissingTipsAsMissing() {
        TreeParser tree = this.createTree();
        Mutations mutations = this.createMutations(tree, new String[] {"-A", "NA", "AA"}, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "X"), 1.5, Mutation.BRANCH_START, 0, A, G)
        });

        assertEquals("-\t0:1@" + this.getNodeNr(tree, "X") + "(2,1.0)", this.logExcess(mutations));
    }

    private TreeParser createTree() {
        TreeParser tree = new TreeParser();
        tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);
        return tree;
    }

    private Mutations createMutations(TreeParser tree, String[] sequences, Mutation[] mutationsToAdd) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", sequences[0]), new Sequence("B", sequences[1]), new Sequence("C", sequences[2])
        ), "dataType", "nucleotide");

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        List<List<Mutation>> history = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            history.add(new ArrayList<>());
        }
        for (Mutation mutation : mutationsToAdd) {
            history.get(mutation.nodeNr()).add(mutation);
        }
        mutations.initialiseMutations(history);

        return mutations;
    }

    private String logExcess(Mutations mutations) {
        ExcessMutationLogger logger = new ExcessMutationLogger();
        logger.initByName("mutations", mutations);

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        logger.log(0, new PrintStream(buffer));
        return buffer.toString().strip();
    }

    private int getNodeNr(TreeParser tree, String taxon) {
        for (Node node : tree.getNodesAsArray()) {
            if (taxon.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Unknown node " + taxon);
    }

}
