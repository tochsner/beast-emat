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

    /* BranchChanges */

    @Test
    public void testBranchChangesAreSortedBySite() {
        BranchChanges changes = BranchChanges.of(3, new int[]{9, 2, 5}, new int[]{0, 1, 2}, new int[]{1, 3, 0});

        assertEquals(3, changes.size());
        assertEquals(2, changes.getSite(0));
        assertEquals(1, changes.getFrom(0));
        assertEquals(3, changes.getTo(0));
        assertEquals(9, changes.getSite(2));
        assertEquals(0, changes.getFrom(2));
        assertEquals(1, changes.getTo(2));

        assertEquals(1, changes.indexOfSite(5));
        assertTrue(changes.indexOfSite(4) < 0);

        assertThrows(IllegalArgumentException.class, () -> BranchChanges.of(3, new int[]{1, 1}, new int[]{0, 0}, new int[]{2, 3}));
        assertThrows(IllegalArgumentException.class, () -> BranchChanges.ofSorted(3, new int[]{4, 1}, new int[]{0, 0}, new int[]{2, 3}, 2));
        assertThrows(IllegalArgumentException.class, () -> BranchChanges.ofSorted(3, new int[]{1}, new int[]{2}, new int[]{2}, 1));
    }

    /* Incremental updates */

    /**
     * Sets random states of nodes and of the root, each in its own accepted proposal, and
     * checks after every one that the node states match the expected sequences and the counts
     * match a computation from the full sequences.
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

    /** Sets random states of a random node or of the root at random sites, and updates the expected sequences. */
    private void applyRandomChange(int[][] expectedSequences) {
        Node[] nodes = this.tree.getNodesAsArray();
        Node node = nodes[this.random.nextInt(nodes.length)];

        int[] sites = this.drawSortedSites(this.random.nextInt(this.nodeStates.getNumSites() + 1));
        int[] states = new int[sites.length];
        for (int i = 0; i < sites.length; i++) {
            states[i] = this.random.nextInt(NUM_STATES);
            expectedSequences[node.getNr()][sites[i]] = states[i];
        }
        this.nodeStates.setStates(node, sites, states, this.operator);
    }

    private int[] drawSortedSites(int numSites) {
        return this.random.ints(0, this.nodeStates.getNumSites()).distinct().limit(numSites).sorted().toArray();
    }

    @Test
    public void testSetStatesRejectsInvalidInput() {
        Node tip = this.getTip("A");
        assertThrows(IllegalArgumentException.class, () -> this.nodeStates.setStates(tip, new int[]{3, 1}, new int[]{0, 0}, this.operator));
        assertThrows(IllegalArgumentException.class, () -> this.nodeStates.setStates(tip, new int[]{1}, new int[]{NUM_STATES}, this.operator));
        assertThrows(IllegalArgumentException.class, () -> this.nodeStates.setStates(tip, new int[]{this.nodeStates.getNumSites()}, new int[]{0}, this.operator));
    }

    /* Lookups */

    @Test
    public void testStateLookupsMatchSequences() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 100; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        for (Node node : this.tree.getNodesAsArray()) {
            assertArrayEquals(expectedSequences[node.getNr()], this.nodeStates.materializeState(node));
            for (int site = 0; site < this.nodeStates.getNumSites(); site++) {
                assertEquals(expectedSequences[node.getNr()][site], this.nodeStates.getState(node, site));
            }
        }
    }

    /** Composes the changes along the path between random pairs of nodes, which can be any two nodes of the tree. */
    @Test
    public void testPairwiseCountsAndChangesMatchSequences() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 200; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        Node[] nodes = this.tree.getNodesAsArray();
        int[] counts = new int[NUM_STATES * NUM_STATES];
        for (Node upper : nodes) {
            for (Node lower : nodes) {
                int[] upperSequence = expectedSequences[upper.getNr()];
                int[] lowerSequence = expectedSequences[lower.getNr()];

                int[] expectedCounts = new int[NUM_STATES * NUM_STATES];
                for (int site = 0; site < upperSequence.length; site++) {
                    expectedCounts[upperSequence[site] * NUM_STATES + lowerSequence[site]]++;
                }

                this.nodeStates.countTransitions(upper, lower, counts);
                assertArrayEquals(expectedCounts, counts, "Counts from " + upper.getNr() + " to " + lower.getNr());

                BranchChanges changes = this.nodeStates.composeChanges(upper, lower);
                BranchChanges expectedChanges = BranchChanges.fromSequences(lower.getNr(), upperSequence, lowerSequence);
                assertTrue(expectedChanges.hasSameEntries(changes), "Changes from " + upper.getNr() + " to " + lower.getNr());
            }
        }
    }

    /** Counts the triples of states of all triples of nodes, which can be any three nodes of the tree, including repeated ones. */
    @Test
    public void testTripleCountsMatchSequences() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 200; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        Node[] nodes = this.tree.getNodesAsArray();
        int[] counts = new int[NUM_STATES * NUM_STATES * NUM_STATES];
        for (Node top : nodes) {
            for (Node bottom : nodes) {
                for (Node side : nodes) {
                    int[] expectedCounts = new int[counts.length];
                    for (int site = 0; site < expectedSequences[top.getNr()].length; site++) {
                        int topState = expectedSequences[top.getNr()][site];
                        int bottomState = expectedSequences[bottom.getNr()][site];
                        int sideState = expectedSequences[side.getNr()][site];
                        expectedCounts[(topState * NUM_STATES + bottomState) * NUM_STATES + sideState]++;
                    }

                    this.nodeStates.countTriples(top, bottom, side, counts);
                    assertArrayEquals(expectedCounts, counts, "Triples of " + top.getNr() + ", " + bottom.getNr() + ", " + side.getNr());
                }
            }
        }
    }

    /* Topology changes */

    /**
     * Swaps two tips between their parents and checks that installing the changes composed
     * before the move keeps all states and recomputes the counts.
     */
    @Test
    public void testSetBranchChangesAfterTopologyChange() {
        int[][] expectedSequences = this.getSequences();
        for (int i = 0; i < 50; i++) {
            this.applyRandomChange(expectedSequences);
        }
        this.nodeStates.setEverythingDirty(false);

        Node tipA = this.getTip("A");
        Node tipC = this.getTip("C");
        Node parentA = tipA.getParent();
        Node parentC = tipC.getParent();

        BranchChanges changesA = this.nodeStates.composeChanges(parentC, tipA);
        BranchChanges changesC = this.nodeStates.composeChanges(parentA, tipC);

        parentA.removeChild(tipA);
        parentC.removeChild(tipC);
        parentA.addChild(tipC);
        parentC.addChild(tipA);

        this.nodeStates.setBranchChanges(tipA, changesA, this.operator);
        this.nodeStates.setBranchChanges(tipC, changesC, this.operator);

        assertSequencesEqual(expectedSequences, this.getSequences());
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

        int newInnerState = (this.nodeStates.getState(inner, 0) + 1) % NUM_STATES;
        this.nodeStates.setStates(inner, new int[]{0}, new int[]{newInnerState}, this.operator);

        Set<Integer> expectedDirty = new HashSet<>(List.of(inner.getNr(), inner.getChild(0).getNr(), inner.getChild(1).getNr()));
        assertEquals(expectedDirty, this.getDirtyNodeNrs());
        this.nodeStates.setEverythingDirty(false);

        Node root = this.tree.getRoot();
        int newRootState = (this.nodeStates.getRootSequence()[3] + 1) % NUM_STATES;
        this.nodeStates.setStates(root, new int[]{3}, new int[]{newRootState}, this.operator);

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

    /** Only tips whose states were set directly are marked for a compatibility check, and every tip after a replacement of all states. */
    @Test
    public void testPendingCompatibilityChecks() {
        // the states are clean after the set up
        assertFalse(this.nodeStates.isEveryCheckPending());
        assertEquals(0, this.nodeStates.getCheckNodeCount());

        Node tipA = this.getTip("A");
        this.nodeStates.setStates(tipA.getParent(), new int[]{0}, new int[]{(this.nodeStates.getState(tipA.getParent(), 0) + 1) % NUM_STATES}, this.operator);
        assertEquals(0, this.nodeStates.getCheckNodeCount());

        this.nodeStates.setStates(tipA, new int[]{1}, new int[]{(this.nodeStates.getState(tipA, 1) + 1) % NUM_STATES}, this.operator);
        assertEquals(1, this.nodeStates.getCheckNodeCount());
        assertEquals(tipA.getNr(), this.nodeStates.getCheckNodeNrs()[0]);
        this.nodeStates.setEverythingDirty(false);
        assertEquals(0, this.nodeStates.getCheckNodeCount());

        this.nodeStates.setBranchChanges(tipA, this.nodeStates.getChanges(tipA), this.operator);
        assertEquals(1, this.nodeStates.getCheckNodeCount());
        this.nodeStates.setEverythingDirty(false);

        int[][] sequences = this.drawSequences();
        this.nodeStates.initialiseStates(sequences[this.tree.getRoot().getNr()], this.getChanges(sequences));
        assertTrue(this.nodeStates.isEveryCheckPending());
    }

    /* Initialisation */

    @Test
    public void testInitialiseStatesFromSequences() {
        int rootNr = this.tree.getRoot().getNr();

        int[][] sequences = this.drawSequences();
        List<BranchChanges> changes = this.getChanges(sequences);
        this.nodeStates.initialiseStates(sequences[rootNr], changes);

        assertSequencesEqual(sequences, this.getSequences());
        this.assertCountsConsistent();

        // the root has no parent to differ from
        List<BranchChanges> invalidChanges = new ArrayList<>(changes);
        invalidChanges.set(rootNr, BranchChanges.fromSequences(rootNr, sequences[0], sequences[rootNr]));
        assertThrows(RuntimeException.class, () -> this.nodeStates.initialiseStates(sequences[rootNr], invalidChanges));
    }

    /** A change must start in the state of the parent at its site. */
    @Test
    public void testInitialiseStatesRejectsInconsistentChanges() {
        int[][] sequences = this.drawSequences();
        int rootNr = this.tree.getRoot().getNr();

        Node tip = this.getTip("A");
        int site = 0;
        int wrongFromState = (sequences[tip.getParent().getNr()][site] + 1) % NUM_STATES;
        int toState = (wrongFromState + 1) % NUM_STATES;

        List<BranchChanges> changes = this.getChanges(sequences);
        changes.set(tip.getNr(), BranchChanges.of(tip.getNr(), new int[]{site}, new int[]{wrongFromState}, new int[]{toState}));
        assertThrows(RuntimeException.class, () -> this.nodeStates.initialiseStates(sequences[rootNr], changes));
    }

    /** Replaces all states within proposals, which a restore must revert and an accept must keep consistent. */
    @Test
    public void testReplaceStatesWithinProposal() {
        int rootNr = this.tree.getRoot().getNr();

        for (int step = 0; step < 100; step++) {
            int[][] sequencesBefore = this.getSequences();
            int[][] countsBefore = this.copyCounts();
            int[] rootStateCountsBefore = this.nodeStates.getRootStateCounts().clone();

            int[][] sequences = this.drawSequences();
            this.nodeStates.replaceStates(sequences[rootNr], this.getChanges(sequences).toArray(new BranchChanges[0]), this.operator);
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

    /** Returns random full sequences for every node. */
    private int[][] drawSequences() {
        int[][] sequences = new int[this.tree.getNodeCount()][this.nodeStates.getNumSites()];
        for (int[] sequence : sequences) {
            for (int site = 0; site < sequence.length; site++) {
                sequence[site] = this.random.nextInt(NUM_STATES);
            }
        }
        return sequences;
    }

    /** Returns the changes of every node relative to its parent, which are empty for the root. */
    private List<BranchChanges> getChanges(int[][] sequences) {
        List<BranchChanges> changes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            changes.add(node.isRoot()
                    ? BranchChanges.empty(nodeNr)
                    : BranchChanges.fromSequences(nodeNr, sequences[node.getParent().getNr()], sequences[nodeNr]));
        }
        return changes;
    }

    /**
     * Replaces the states at a random subset of sites within a proposal, so that the root
     * states change at some of them and only some nodes change. The states must equal the
     * expected sequences, the counts must match a computation from scratch, every node whose
     * states change must be dirty, and a restore must bring back the old states.
     */
    @Test
    public void testReplaceStatesAtSitesWithinProposal() {
        int numSites = this.nodeStates.getNumSites();
        int rootNr = this.tree.getRoot().getNr();

        for (int step = 0; step < 200; step++) {
            int[][] sequencesBefore = this.getSequences();
            int[][] countsBefore = this.copyCounts();
            int[] rootStateCountsBefore = this.nodeStates.getRootStateCounts().clone();

            // draw ascending sites, and new states for a random subset of the nodes at them

            int[] sites = this.drawSortedSites(1 + this.random.nextInt(numSites));
            int[][] sequences = new int[sequencesBefore.length][];
            for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
                sequences[nodeNr] = sequencesBefore[nodeNr].clone();
                if (this.random.nextInt(3) == 0) {
                    for (int site : sites) {
                        sequences[nodeNr][site] = this.random.nextInt(NUM_STATES);
                    }
                }
            }

            int[] rootStates = new int[sites.length];
            for (int i = 0; i < sites.length; i++) {
                rootStates[i] = sequences[rootNr][sites[i]];
            }

            // the changes at the sites relative to the new states of the parent
            BranchChanges[] changesAtSites = new BranchChanges[sequences.length];
            for (Node node : this.tree.getNodesAsArray()) {
                if (node.isRoot()) {
                    changesAtSites[node.getNr()] = BranchChanges.empty(node.getNr());
                    continue;
                }

                int[] parentSequence = sequences[node.getParent().getNr()];
                int[] changedSites = new int[sites.length];
                int[] fromStates = new int[sites.length];
                int[] toStates = new int[sites.length];
                int size = 0;
                for (int site : sites) {
                    if (sequences[node.getNr()][site] != parentSequence[site]) {
                        changedSites[size] = site;
                        fromStates[size] = parentSequence[site];
                        toStates[size] = sequences[node.getNr()][site];
                        size++;
                    }
                }
                changesAtSites[node.getNr()] = BranchChanges.ofSorted(node.getNr(), changedSites, fromStates, toStates, size);
            }

            this.nodeStates.replaceStatesAtSites(sites, rootStates, changesAtSites, this.operator);
            assertSequencesEqual(sequences, this.getSequences());
            this.assertCountsConsistent();
            for (Node node : this.tree.getNodesAsArray()) {
                if (!Arrays.equals(sequencesBefore[node.getNr()], sequences[node.getNr()])) {
                    assertTrue(this.nodeStates.isDirty(node), "node " + node.getNr() + " changed but is clean");
                }
            }
            assertEquals(0, this.nodeStates.getCheckNodeCount());

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

    /** Changes at a site that is not replaced, or unsorted sites, are rejected. */
    @Test
    public void testReplaceStatesAtSitesRejectsInvalidInput() {
        int numNodes = this.tree.getNodeCount();
        // BEAST numbers the leaves first
        int leafNr = 0;
        int[] root = this.nodeStates.getRootSequence();

        BranchChanges[] changes = new BranchChanges[numNodes];
        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            changes[nodeNr] = BranchChanges.empty(nodeNr);
        }
        changes[leafNr] = BranchChanges.of(leafNr, new int[]{2}, new int[]{root[2]}, new int[]{(root[2] + 1) % NUM_STATES});
        BranchChanges[] outsideChanges = changes;
        assertThrows(IllegalArgumentException.class,
                () -> this.nodeStates.replaceStatesAtSites(new int[]{1}, new int[]{root[1]}, outsideChanges, this.operator));

        changes[leafNr] = BranchChanges.empty(leafNr);
        assertThrows(IllegalArgumentException.class,
                () -> this.nodeStates.replaceStatesAtSites(new int[]{3, 1}, new int[]{root[3], root[1]}, changes, this.operator));
    }

    /* Brute force reference */

    private int[][] getSequences() {
        return getSequences(this.nodeStates, this.tree);
    }

    /** Returns the full sequence of every node, as visited from the root down. */
    private static int[][] getSequences(NodeStates nodeStates, Tree tree) {
        int[][] sequences = new int[tree.getNodeCount()][];
        nodeStates.visitStates((node, sequence) -> sequences[node.getNr()] = sequence.clone());
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
        assertArrayEquals(rootSequence, this.nodeStates.getRootSequence());

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

            int[] expectedStateCounts = new int[NUM_STATES];
            for (int state : nodeSequence) {
                expectedStateCounts[state]++;
            }
            assertArrayEquals(expectedStateCounts, this.nodeStates.getStateCounts(node), "State counts of node " + node.getNr());
        }

        assertEquals(expectedDifferenceCount, this.nodeStates.getDifferenceCount());
    }

    private static void assertSequencesEqual(int[][] expected, int[][] actual) {
        assertTrue(Arrays.deepEquals(expected, actual), "Node sequences differ.");
    }

}
