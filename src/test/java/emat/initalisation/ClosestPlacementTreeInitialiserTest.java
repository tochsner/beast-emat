package emat.initalisation;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ClosestPlacementTreeInitialiserTest {

    // two clades, {A, B, C} and {D, E, F}, where each later tip has one difference to the previous one
    private static final String[] NAMES = {"A", "B", "C", "D", "E", "F"};
    private static final String[] SEQUENCES = {
            "AAAAAAAAAACCCCCCCCCC",
            "AAAAAAAAAACCCCCCCCCA",
            "AAAAAAAAAACCCCCCCCGA",
            "GGGGGGGGGGTTTTTTTTTT",
            "GGGGGGGGGGTTTTTTTTTA",
            "GGGGGGGGGGTTTTTTTTCA"
    };
    private static final String DATES = "A=2000,B=2001,C=2002,D=2000,E=2003,F=2004";

    /** Checks that the tips keep their dates and that every branch has at least the minimum length. */
    @Test
    public void testTreeIsValidAndDated() {
        Tree tree = this.createTree(this.createAlignment(), DATES);
        this.initialiseTree(tree, 0.01);

        assertEquals(2 * NAMES.length - 1, tree.getNodeCount());
        for (Node tip : tree.getExternalNodes()) {
            double expectedHeight = tree.getDateTrait().getValue(tip.getID());
            assertEquals(expectedHeight, tip.getHeight(), 1e-12);
        }
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                double branchLength = node.getParent().getHeight() - node.getHeight();
                assertTrue(branchLength >= ClosestPlacementTreeInitialiser.MIN_BRANCH_LENGTH - 1e-12);
            }
        }
    }

    /**
     * Checks that the last tip of each clade becomes the sibling of its closest tip and that the
     * two clades are separated at the root.
     */
    @Test
    public void testClosestTipsAreSiblings() {
        Tree tree = this.createTree(this.createAlignment(), DATES);
        this.initialiseTree(tree, 0.01);

        assertSame(this.getTip(tree, "B").getParent(), this.getTip(tree, "C").getParent());
        assertSame(this.getTip(tree, "E").getParent(), this.getTip(tree, "F").getParent());

        Node root = tree.getRoot();
        assertEquals(3, root.getLeft().getLeafNodeCount());
        assertEquals(3, root.getRight().getLeafNodeCount());
    }

    /** Checks that without dates all tips are at height zero and the tree is still valid. */
    @Test
    public void testTreeWithoutDates() {
        Tree tree = this.createTree(this.createAlignment(), null);
        this.initialiseTree(tree, 0.01);

        for (Node tip : tree.getExternalNodes()) {
            assertEquals(0.0, tip.getHeight());
        }
        for (Node node : tree.getInternalNodes()) {
            assertTrue(node.getHeight() > Math.max(node.getLeft().getHeight(), node.getRight().getHeight()));
        }
    }

    /**
     * Checks the clock rate of tips sampled one year apart, where every tip has one more
     * difference to the oldest tip than the previous one, at a rate of 0.01 per site and year.
     */
    @Test
    public void testClockRateOfLadder() {
        int numTips = 10;
        List<Sequence> sequences = new ArrayList<>();
        StringBuilder dates = new StringBuilder();
        for (int i = 0; i < numTips; i++) {
            char[] sequence = "A".repeat(100).toCharArray();
            for (int site = 0; site < i; site++) {
                sequence[site] = 'C';
            }
            sequences.add(new Sequence("t" + i, new String(sequence)));
            dates.append(i == 0 ? "" : ",").append("t").append(i).append("=").append(2000 + i);
        }
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");
        Tree tree = this.createTree(alignment, dates.toString());

        RealScalarParam<?> clockRate = this.initialiseClockRate(tree, alignment, 1.0);

        assertEquals(0.01, clockRate.get(), 1e-12);
    }

    /** Checks that contemporaneous tips keep the given clock rate. */
    @Test
    public void testClockRateOfContemporaneousTipsIsKept() {
        Alignment alignment = this.createAlignment();
        Tree tree = this.createTree(alignment, "A=2000,B=2000,C=2000,D=2000,E=2000,F=2000");

        RealScalarParam<?> clockRate = this.initialiseClockRate(tree, alignment, 0.5);

        assertEquals(0.5, clockRate.get());
    }

    private Alignment createAlignment() {
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < NAMES.length; i++) {
            sequences.add(new Sequence(NAMES[i], SEQUENCES[i]));
        }

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");
        return alignment;
    }

    /** Returns a tree on the taxa of the alignment, with a date trait if dates are given. */
    private Tree createTree(Alignment alignment, String dates) {
        TaxonSet taxonSet = new TaxonSet();
        taxonSet.initByName("alignment", alignment);

        Tree tree = new Tree();
        if (dates == null) {
            tree.initByName("taxonset", taxonSet);
            return tree;
        }

        TraitSet dateTrait = new TraitSet();
        dateTrait.initByName("traitname", "date", "taxa", taxonSet, "value", dates);
        tree.initByName("taxonset", taxonSet, "trait", dateTrait);
        return tree;
    }

    private void initialiseTree(Tree tree, double clockRate) {
        ClosestPlacementTreeInitialiser initialiser = new ClosestPlacementTreeInitialiser();
        initialiser.initByName(
                "tree", tree,
                "alignment", this.createAlignment(),
                "clockRate", new RealScalarParam<>(clockRate, PositiveReal.INSTANCE)
        );
        initialiser.initStateNodes();
    }

    private RealScalarParam<?> initialiseClockRate(Tree tree, Alignment alignment, double initialValue) {
        RealScalarParam<?> clockRate = new RealScalarParam<>(initialValue, PositiveReal.INSTANCE);
        ClockRateInitialiser initialiser = new ClockRateInitialiser();
        initialiser.initByName("clockRate", clockRate, "tree", tree, "alignment", alignment);
        initialiser.initStateNodes();
        return clockRate;
    }

    private Node getTip(Tree tree, String name) {
        for (Node tip : tree.getExternalNodes()) {
            if (tip.getID().equals(name)) {
                return tip;
            }
        }
        throw new IllegalArgumentException("No tip " + name);
    }

}
