package esat.state;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NodeStatesTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";
    private static final int NUM_STATES = 4;

    private Tree tree;
    private NodeStates nodeStates;
    private Operator operator;
    private Random random;

    @BeforeEach
    public void setUp() {
        this.tree = new TreeParser();
        this.tree.initByName(
                "newick", NEWICK,
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "ACGTACGTACGT"),
                        new Sequence("B", "ACNAACGNNCGT"),
                        new Sequence("C", "TCGAACCTACGA"),
                        new Sequence("D", "TCGTACCTAGGA"),
                        new Sequence("E", "GCGTTNNNNNGT"),
                        new Sequence("F", "GCGTTCGAACTT")
                ),
                "dataType", "nucleotide"
        );

        this.nodeStates = new NodeStates();
        this.nodeStates.initByName("tree", this.tree, "alignment", alignment);
        this.nodeStates.setEverythingDirty(false);

        this.operator = new Operator() {
            @Override
            public void initAndValidate() {
            }

            @Override
            public double proposal() {
                return 0.0;
            }
        };
        this.random = new Random(7);
    }

    /* StateChanges */

    @Test
    public void testStateChangesAreSortedBySite() {
        StateChanges changes = StateChanges.of(3, new int[]{9, 2, 5}, new int[]{1, 3, 0});

        assertEquals(3, changes.size());
        assertEquals(2, changes.getSite(0));
        assertEquals(3, changes.getState(0));
        assertEquals(9, changes.getSite(2));
        assertEquals(1, changes.getState(2));

        assertEquals(1, changes.indexOfSite(5));
        assertTrue(changes.indexOfSite(4) < 0);
        assertEquals(0, changes.getStateOrDefault(5, 2));
        assertEquals(2, changes.getStateOrDefault(4, 2));

        assertThrows(IllegalArgumentException.class, () -> StateChanges.of(3, new int[]{1, 1}, new int[]{0, 2}));
        assertThrows(IllegalArgumentException.class, () -> StateChanges.ofSorted(3, new int[]{4, 1}, new int[]{0, 2}, 2));
    }

    /* Incremental updates */

    /**
     * Applies random changes of nodes and of the root sequence, each in its own accepted
     * proposal, and checks after every one that the node states match the expected sequences
     * and the counts match a computation from the full sequences.
     */
    @Test
    public void testRandomChangesKeepCountsConsistent() {
        int[][] expectedSequences = this.getSequences();

        for (int step = 0; step < 2000; step++) {
            this.applyRandomChange(expectedSequences);
            this.nodeStates.setEverythingDirty(false);

            assertSequencesEqual(expectedSequences, this.getSequences());
            this.assertCountsConsistent();
        }
    }

    /**
     * Runs proposals of several random changes, which are accepted or rejected at random. A
     * rejected proposal must restore the exact states from before it, even if the states are
     * stored and restored a second time like a calculation node downstream of the tree.
     */
    @Test
    public void testRestoreRevertsProposal() {
        int[][] expectedSequences = this.getSequences();

        for (int step = 0; step < 1000; step++) {
            int[][] sequencesBefore = this.getSequences();
            int[][] countsBefore = this.copyCounts();
            int[] rootStateCountsBefore = this.nodeStates.getRootStateCounts().clone();

            int numChanges = 1 + this.random.nextInt(4);
            for (int i = 0; i < numChanges; i++) {
                this.applyRandomChange(expectedSequences);
            }

            // the second store of a calculation node must not overwrite the snapshot
            this.nodeStates.store();

            if (this.random.nextBoolean()) {
                this.nodeStates.restoreStateNode();
                this.nodeStates.restore();

                assertSequencesEqual(sequencesBefore, this.getSequences());
                assertArrayEquals(rootStateCountsBefore, this.nodeStates.getRootStateCounts());
                assertTrue(Arrays.deepEquals(countsBefore, this.copyCounts()));
                expectedSequences = sequencesBefore;
            } else {
                assertSequencesEqual(expectedSequences, this.getSequences());
            }

            this.nodeStates.setEverythingDirty(false);
            this.assertCountsConsistent();
        }
    }

    /** Changes the states of a random non-root node or the root sequence, and updates the expected sequences. */
    private void applyRandomChange(int[][] expectedSequences) {
        Node[] nodes = this.tree.getNodesAsArray();
        Node node = nodes[this.random.nextInt(nodes.length)];
        int numSites = this.nodeStates.getNumSites();

        if (node.isRoot()) {
            int numChangedSites = this.random.nextInt(numSites + 1);
            int[] sites = this.drawDistinctSites(numChangedSites);
            int[] states = new int[numChangedSites];
            for (int i = 0; i < numChangedSites; i++) {
                states[i] = this.random.nextInt(NUM_STATES);
                expectedSequences[node.getNr()][sites[i]] = states[i];
            }
            this.nodeStates.applyRootStates(sites, states, this.operator);
        } else {
            int[] sequence = expectedSequences[node.getNr()];
            for (int site : this.drawDistinctSites(this.random.nextInt(numSites + 1))) {
                sequence[site] = this.random.nextInt(NUM_STATES);
            }
            StateChanges changes = StateChanges.fromSequence(node.getNr(), sequence, this.nodeStates.getReferenceSequence());
            this.nodeStates.applyChanges(node, changes, this.operator);
        }
    }

    private int[] drawDistinctSites(int numSites) {
        List<Integer> sites = new ArrayList<>();
        for (int site = 0; site < this.nodeStates.getNumSites(); site++) {
            sites.add(site);
        }
        Collections.shuffle(sites, this.random);
        return sites.subList(0, numSites).stream().mapToInt(Integer::intValue).toArray();
    }

    /* Topology changes */

    /** Swaps two tips between their parents and checks that marking their branches as changed recomputes the counts. */
    @Test
    public void testMarkBranchChangedAfterTopologyChange() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 50; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        Node tipA = this.getTip("A");
        Node tipC = this.getTip("C");
        Node parentA = tipA.getParent();
        Node parentC = tipC.getParent();

        parentA.removeChild(tipA);
        parentC.removeChild(tipC);
        parentA.addChild(tipC);
        parentC.addChild(tipA);

        this.nodeStates.markBranchChanged(tipA, this.operator);
        this.nodeStates.markBranchChanged(tipC, this.operator);

        this.assertCountsConsistent();
        assertTrue(this.nodeStates.isDirty(tipA));
        assertTrue(this.nodeStates.isDirty(tipC));
    }

    private Node getTip(String taxon) {
        for (Node node : this.tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node;
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon);
    }

    /* Copies */

    @Test
    public void testCopyIsIndependentOfOriginal() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 20; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        NodeStates copy = this.nodeStates.copy();
        int[][] copiedSequences = getSequences(copy, this.tree);
        int[][] copiedCounts = copyCounts(copy, this.tree);

        for (int i = 0; i < 20; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        assertSequencesEqual(copiedSequences, getSequences(copy, this.tree));
        assertTrue(Arrays.deepEquals(copiedCounts, copyCounts(copy, this.tree)));
    }

    /* Dirty tracking */

    @Test
    public void testDirtyNodesOfChangeAndRootChange() {
        Node tipA = this.getTip("A");
        Node inner = tipA.getParent();

        int[] sequence = getSequences(this.nodeStates, this.tree)[inner.getNr()];
        sequence[0] = (sequence[0] + 1) % NUM_STATES;
        this.nodeStates.applyChanges(inner, StateChanges.fromSequence(inner.getNr(), sequence, this.nodeStates.getReferenceSequence()), this.operator);

        Set<Integer> expectedDirty = new HashSet<>(List.of(inner.getNr(), inner.getChild(0).getNr(), inner.getChild(1).getNr()));
        assertEquals(expectedDirty, this.getDirtyNodeNrs());
        this.nodeStates.setEverythingDirty(false);

        Node root = this.tree.getRoot();
        int newRootState = (this.nodeStates.getReferenceSequence()[3] + 1) % NUM_STATES;
        this.nodeStates.applyRootStates(new int[]{3}, new int[]{newRootState}, this.operator);

        expectedDirty = new HashSet<>(List.of(root.getNr(), root.getChild(0).getNr(), root.getChild(1).getNr()));
        assertEquals(expectedDirty, this.getDirtyNodeNrs());
        this.nodeStates.setEverythingDirty(false);

        assertTrue(this.getDirtyNodeNrs().isEmpty());
        assertFalse(this.nodeStates.isDirty(root));
    }

    private Set<Integer> getDirtyNodeNrs() {
        assertFalse(this.nodeStates.isEveryBranchDirty());
        Set<Integer> nodeNrs = new HashSet<>();
        for (int i = 0; i < this.nodeStates.getDirtyNodeCount(); i++) {
            nodeNrs.add(this.nodeStates.getDirtyNodeNrs()[i]);
        }
        for (Node node : this.tree.getNodesAsArray()) {
            assertEquals(nodeNrs.contains(node.getNr()), this.nodeStates.isDirty(node));
        }
        return nodeNrs;
    }

    /* Initialisation */

    @Test
    public void testInitialiseStatesFromSequences() {
        int numSites = this.nodeStates.getNumSites();
        int rootNr = this.tree.getRoot().getNr();

        int[][] sequences = new int[this.tree.getNodeCount()][numSites];
        for (int[] sequence : sequences) {
            for (int site = 0; site < numSites; site++) {
                sequence[site] = this.random.nextInt(NUM_STATES);
            }
        }

        List<StateChanges> changes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
            changes.add(StateChanges.fromSequence(nodeNr, sequences[nodeNr], sequences[rootNr]));
        }
        this.nodeStates.initialiseStates(sequences[rootNr], changes);

        assertSequencesEqual(sequences, this.getSequences());
        this.assertCountsConsistent();

        // the root must not differ from the root sequence
        List<StateChanges> invalidChanges = new ArrayList<>(changes);
        invalidChanges.set(rootNr, StateChanges.fromSequence(rootNr, sequences[0], sequences[rootNr]));
        assertThrows(RuntimeException.class, () -> this.nodeStates.initialiseStates(sequences[rootNr], invalidChanges));
    }

    /** Replaces all states within proposals, which a restore must revert and an accept must keep consistent. */
    @Test
    public void testReplaceStatesWithinProposal() {
        int numSites = this.nodeStates.getNumSites();
        int rootNr = this.tree.getRoot().getNr();

        for (int step = 0; step < 100; step++) {
            int[][] sequencesBefore = this.getSequences();
            int[][] countsBefore = this.copyCounts();
            int[] rootStateCountsBefore = this.nodeStates.getRootStateCounts().clone();

            int[][] sequences = new int[this.tree.getNodeCount()][numSites];
            for (int[] sequence : sequences) {
                for (int site = 0; site < numSites; site++) {
                    sequence[site] = this.random.nextInt(NUM_STATES);
                }
            }
            StateChanges[] changes = new StateChanges[sequences.length];
            for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
                changes[nodeNr] = StateChanges.fromSequence(nodeNr, sequences[nodeNr], sequences[rootNr]);
            }

            this.nodeStates.replaceStates(sequences[rootNr], changes, this.operator);
            assertTrue(this.nodeStates.isEveryBranchDirty());
            assertSequencesEqual(sequences, this.getSequences());
            this.assertCountsConsistent();

            if (this.random.nextBoolean()) {
                this.nodeStates.restoreStateNode();
                this.nodeStates.restore();

                assertSequencesEqual(sequencesBefore, this.getSequences());
                assertArrayEquals(rootStateCountsBefore, this.nodeStates.getRootStateCounts());
                assertTrue(Arrays.deepEquals(countsBefore, this.copyCounts()));
            }

            this.nodeStates.setEverythingDirty(false);
            this.assertCountsConsistent();
        }
    }

    /* Brute force reference */

    private int[][] getSequences() {
        return getSequences(this.nodeStates, this.tree);
    }

    /** Returns the full sequence of every node, built from the root sequence and the changes of the node. */
    private static int[][] getSequences(NodeStates nodeStates, Tree tree) {
        int[][] sequences = new int[tree.getNodeCount()][];
        for (Node node : tree.getNodesAsArray()) {
            int[] sequence = nodeStates.getReferenceSequence().clone();
            StateChanges changes = nodeStates.getChanges(node);
            for (int i = 0; i < changes.size(); i++) {
                assertTrue(changes.getState(i) != nodeStates.getReferenceSequence()[changes.getSite(i)]);
                sequence[changes.getSite(i)] = changes.getState(i);
            }
            sequences[node.getNr()] = sequence;
        }
        return sequences;
    }

    private int[][] copyCounts() {
        return copyCounts(this.nodeStates, this.tree);
    }

    /** Returns a copy of the transition counts of every non-root node, and null for the root. */
    private static int[][] copyCounts(NodeStates nodeStates, Tree tree) {
        int[][] counts = new int[tree.getNodeCount()][];
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                counts[node.getNr()] = nodeStates.getTransitionCounts(node).clone();
            }
        }
        return counts;
    }

    /** Checks the root state counts and the transition counts of every branch against counts of the full sequences. */
    private void assertCountsConsistent() {
        int[][] sequences = this.getSequences();
        int[] rootSequence = sequences[this.tree.getRoot().getNr()];

        int[] expectedRootStateCounts = new int[NUM_STATES];
        for (int state : rootSequence) {
            expectedRootStateCounts[state]++;
        }
        assertArrayEquals(expectedRootStateCounts, this.nodeStates.getRootStateCounts());
        assertArrayEquals(rootSequence, this.nodeStates.getReferenceSequence());

        int expectedDifferenceCount = 0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            int[] parentSequence = sequences[node.getParent().getNr()];
            int[] nodeSequence = sequences[node.getNr()];
            int[] expectedCounts = new int[NUM_STATES * NUM_STATES];
            for (int site = 0; site < nodeSequence.length; site++) {
                expectedCounts[parentSequence[site] * NUM_STATES + nodeSequence[site]]++;
                if (parentSequence[site] != nodeSequence[site]) {
                    expectedDifferenceCount++;
                }
            }

            assertArrayEquals(expectedCounts, this.nodeStates.getTransitionCounts(node), "Counts above node " + node.getNr());
        }

        assertEquals(expectedDifferenceCount, this.nodeStates.getDifferenceCount());

        int expectedChangeCount = 0;
        for (int[] sequence : sequences) {
            for (int site = 0; site < sequence.length; site++) {
                if (sequence[site] != rootSequence[site]) {
                    expectedChangeCount++;
                }
            }
        }
        assertEquals(expectedChangeCount, this.nodeStates.getChangeCount());
    }

    private static void assertSequencesEqual(int[][] expected, int[][] actual) {
        assertTrue(Arrays.deepEquals(expected, actual), "Node sequences differ.");
    }

}
