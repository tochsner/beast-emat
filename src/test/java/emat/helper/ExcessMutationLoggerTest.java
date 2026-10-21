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

    /**
     * On the tree ((A,B)X,C)R, site 0 is A at all tips and site 1 is G only at C. Site 1 needs
     * a single mutation, so only the A -> C -> A detour on the branch above A at site 0 is
     * in excess.
     */
    @Test
    public void testLogsOnlyMutationsBeyondParsimony() {
        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0)X:1.0,C:2.0)R;", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = this.createMutations(tree, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "A"), 0.6, 1.0, 0, A, C),
                new Mutation(this.getNodeNr(tree, "A"), 0.3, 0.6, 0, C, A),
                new Mutation(this.getNodeNr(tree, "C"), 0.5, 2.0, 1, A, G)
        });

        int nodeNr = this.getNodeNr(tree, "A");
        assertEquals("0:2@" + nodeNr + "(1,0.0)," + nodeNr + "(1,0.0)", this.logExcess(mutations));
    }

    @Test
    public void testLogsPlaceholderWithoutExcess() {
        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0)X:1.0,C:2.0)R;", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = this.createMutations(tree, new Mutation[] {
                new Mutation(this.getNodeNr(tree, "C"), 0.5, 2.0, 1, A, G)
        });

        assertEquals("-", this.logExcess(mutations));
    }

    private Mutations createMutations(TreeParser tree, Mutation[] mutationsToAdd) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "AA"), new Sequence("B", "AA"), new Sequence("C", "AG")
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
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon);
    }

}
