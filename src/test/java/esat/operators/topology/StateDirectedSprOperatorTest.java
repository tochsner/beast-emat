package esat.operators.topology;

import beast.base.core.Input;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import esat.state.NodeStates;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class StateDirectedSprOperatorTest {

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

    // a small tree on few sites, whose posterior spreads over several topologies
    private static final String SMALL_NEWICK = "(((A:0.2,B:0.2):0.2,C:0.4):0.3,(D:0.3,E:0.3):0.4);";
    private static final List<Sequence> SMALL_SEQUENCES = List.of(
            new Sequence("A", "ACGT"),
            new Sequence("B", "ACGA"),
            new Sequence("C", "ATGT"),
            new Sequence("D", "GTCT"),
            new Sequence("E", "GTNT")
    );

    /* Consistency */

    @Test
    public void testProposalsKeepPriorConsistentOnTree() {
        this.runConsistencyChain(EsatFixture.create(NEWICK, SEQUENCES));
    }

    @Test
    public void testProposalsKeepPriorConsistentOnIncrementalStoreTree() {
        this.runConsistencyChain(EsatFixture.create(EsatFixture.parseIncrementalTree(NEWICK), SEQUENCES));
    }

    /**
     * Drives the operator through the store, accept and restore cycle of MCMC, accepting at
     * random. After every step, the transition counts must match the states and the
     * incremental prior a computation from scratch, and a restore must bring back the tree.
     */
    private void runConsistencyChain(EsatFixture fixture) {
        Randomizer.setSeed(11);
        this.initialiseSampledStates(fixture);
        Tree tree = fixture.tree();

        // a low annealing power spreads the proposals over more branches
        StateDirectedSprOperator operator = this.createOperator(fixture, 0.2, 0.3, false);
        State state = this.createState(fixture);
        double logP = fixture.geneticPrior().calculateLogP();
        state.setEverythingDirty(false);

        int numTopologyChanges = 0;
        for (long i = 0; i < 3000; i++) {
            String newickBefore = tree.getRoot().toNewick();
            String topologyBefore = this.getTopology(tree);
            state.store(i);

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
                double newLogP = fixture.geneticPrior().calculateLogP();

                assertEquals(fixture.computeLogPFromScratch(), newLogP, 1e-9);
                this.assertCountsMatchStates(fixture);
                if (!this.getTopology(tree).equals(topologyBefore)) {
                    numTopologyChanges++;
                }

                if (Randomizer.nextBoolean()) {
                    state.acceptCalculationNodes();
                    logP = newLogP;
                    state.setEverythingDirty(false);
                    continue;
                }
            }

            state.restore();
            state.restoreCalculationNodes();
            state.setEverythingDirty(false);

            assertEquals(newickBefore, tree.getRoot().toNewick());
            assertEquals(logP, fixture.geneticPrior().getCurrentLogP(), 1e-9);
            this.assertCountsMatchStates(fixture);
        }

        assertTrue(numTopologyChanges > 50, "Only " + numTopologyChanges + " proposals changed the topology.");
    }

    private void assertCountsMatchStates(EsatFixture fixture) {
        NodeStates nodeStates = fixture.nodeStates();
        int[] counts = new int[EsatFixture.NUM_STATES * EsatFixture.NUM_STATES];
        for (Node node : fixture.tree().getNodesAsArray()) {
            if (!node.isRoot()) {
                nodeStates.countTransitions(node.getParent(), node, counts);
                assertArrayEquals(counts, nodeStates.getTransitionCounts(node), "Counts above node " + node.getNr());
            }
        }
    }

    /* Posterior */

    /**
     * Compares the operator with a uniform SPR on the genetic prior alone with fixed states.
     * Both chains move the topology and the heights of the inner nodes other than the root,
     * so they must agree on the mean tree length and the frequencies of the most frequent
     * topologies, within the Monte Carlo error from batch means.
     */
    @Test
    public void testAgreesWithUniformSpr() {
        int numSteps = 200000;

        Randomizer.setSeed(12);
        EsatFixture fixture = EsatFixture.create(SMALL_NEWICK, SMALL_SEQUENCES);
        this.initialiseSampledStates(fixture);

        UniformSprOperator uniformOperator = new UniformSprOperator();
        uniformOperator.initByName("tree", fixture.tree(), "nodeStates", fixture.nodeStates(), "weight", 1.0);
        Chain referenceChain = this.runPosteriorChain(fixture, uniformOperator, numSteps, 14);
        Chain operatorChain = this.runPosteriorChain(fixture, this.createOperator(fixture, 0.01, 0.8, false), numSteps, 13);

        this.assertMeansAgree(referenceChain.treeLengths(), operatorChain.treeLengths(), "tree length");
        for (String topology : this.findMostFrequentTopologies(referenceChain, 3)) {
            this.assertMeansAgree(
                    this.indicate(referenceChain, topology), this.indicate(operatorChain, topology), "frequency of " + topology
            );
        }
    }

    /** The tree length and the topology after every step of a chain. */
    private record Chain(double[] treeLengths, String[] topologies) {
    }

    /** Runs a Metropolis-Hastings chain with the given operator on the genetic prior, starting from and leaving the current tree. */
    private Chain runPosteriorChain(EsatFixture fixture, Operator operator, int numSteps, long seed) {
        Tree tree = fixture.tree();
        State state = this.createState(fixture);
        double logP = fixture.geneticPrior().calculateLogP();
        state.setEverythingDirty(false);

        double[] treeLengths = new double[numSteps];
        String[] topologies = new String[numSteps];

        Randomizer.setSeed(seed);
        for (int i = 0; i < numSteps; i++) {
            state.store(i);
            double logHastingsRatio = operator.proposal();

            boolean isAccepted = false;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
                double newLogP = fixture.geneticPrior().calculateLogP();
                isAccepted = Math.log(Randomizer.nextDouble()) < newLogP - logP + logHastingsRatio;
                if (isAccepted) {
                    state.acceptCalculationNodes();
                    logP = newLogP;
                }
            }
            if (!isAccepted) {
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);

            treeLengths[i] = this.computeTreeLength(tree);
            topologies[i] = this.getTopology(tree);
        }

        return new Chain(treeLengths, topologies);
    }

    private List<String> findMostFrequentTopologies(Chain chain, int numTopologies) {
        Map<String, Integer> counts = new HashMap<>();
        for (String topology : chain.topologies()) {
            counts.merge(topology, 1, Integer::sum);
        }
        return counts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(numTopologies)
                .map(Map.Entry::getKey)
                .toList();
    }

    private double[] indicate(Chain chain, String topology) {
        double[] indicators = new double[chain.topologies().length];
        for (int i = 0; i < indicators.length; i++) {
            indicators[i] = chain.topologies()[i].equals(topology) ? 1.0 : 0.0;
        }
        return indicators;
    }

    /** Checks that the means of two chains differ by less than four standard errors, each estimated from 100 batch means. */
    private void assertMeansAgree(double[] referenceValues, double[] values, String description) {
        double[] reference = this.computeMeanAndStandardError(referenceValues);
        double[] estimate = this.computeMeanAndStandardError(values);
        double standardError = Math.sqrt(reference[1] * reference[1] + estimate[1] * estimate[1]);

        double z = Math.abs(estimate[0] - reference[0]) / standardError;
        assertTrue(z < 4.0, "Mean " + description + " " + estimate[0] + " differs from reference " + reference[0] + " by z = " + z);
    }

    private double[] computeMeanAndStandardError(double[] values) {
        int numBatches = 100;
        int batchSize = values.length / numBatches;

        double[] batchMeans = new double[numBatches];
        for (int batch = 0; batch < numBatches; batch++) {
            for (int i = batch * batchSize; i < (batch + 1) * batchSize; i++) {
                batchMeans[batch] += values[i] / batchSize;
            }
        }

        double mean = Arrays.stream(batchMeans).average().orElseThrow();
        double variance = Arrays.stream(batchMeans).map(m -> (m - mean) * (m - mean)).sum() / (numBatches - 1);
        return new double[]{mean, Math.sqrt(variance / numBatches)};
    }

    /** Returns the clades of the inner nodes other than the root, sorted, as a key of the rooted topology. */
    private String getTopology(Tree tree) {
        TreeSet<String> clades = new TreeSet<>();
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf() && !node.isRoot()) {
                TreeSet<String> taxa = new TreeSet<>();
                for (Node tip : node.getAllLeafNodes()) {
                    taxa.add(tip.getID());
                }
                clades.add(String.join("", taxa));
            }
        }
        return String.join("|", clades);
    }

    private double computeTreeLength(Tree tree) {
        double treeLength = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                treeLength += node.getParent().getHeight() - node.getHeight();
            }
        }
        return treeLength;
    }

    /* Weight */

    @Test
    public void testWeightScalesWithSqrtNumTips() {
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        assertEquals(1.0, this.createOperator(fixture, 0.01, 0.8, false).getWeight(), 1e-12);
        assertEquals(Math.sqrt(6.0), this.createOperator(fixture, 0.01, 0.8, true).getWeight(), 1e-12);
    }

    /* Fixtures */

    private StateDirectedSprOperator createOperator(EsatFixture fixture, double fullExplorationProbability, double annealing, boolean scaleWeight) {
        StateDirectedSprOperator operator = new StateDirectedSprOperator();
        operator.initByName(
                "tree", fixture.tree(),
                "nodeStates", fixture.nodeStates(),
                "geneticPrior", fixture.geneticPrior(),
                "fullExplorationProbability", fullExplorationProbability,
                "annealing", annealing,
                "scaleWeightBySqrtNumTips", scaleWeight,
                "weight", 1.0
        );
        return operator;
    }

    private State createState(EsatFixture fixture) {
        State state = new State();
        state.initByName("stateNode", fixture.tree(), "stateNode", fixture.nodeStates());
        state.initialise();
        state.setPosterior(fixture.geneticPrior());
        state.setEverythingDirty(true);
        return state;
    }

    private void initialiseSampledStates(EsatFixture fixture) {
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler.Sample sample = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess).sample();
        fixture.nodeStates().initialiseStates(sample.rootSequence(), List.of(sample.changesAtNode()));
    }

    /**
     * Prunes a uniformly chosen subtree whose parent is not the root and regrafts it with its
     * parent on a uniformly chosen branch of the pruned tree whose top is older than the
     * subtree, at a uniform height. The numbers of subtrees and branches do not change under
     * the move, so the Hastings ratio is the ratio of the lengths of the new and old ranges.
     */
    public static class UniformSprOperator extends TreeOperator {

        final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "", Input.Validate.REQUIRED);

        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            Tree tree = this.treeInput.get();

            List<Node> subtrees = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                if (!node.isRoot() && !node.getParent().isRoot()) {
                    subtrees.add(node);
                }
            }
            Node x = subtrees.get(Randomizer.nextInt(subtrees.size()));
            Node parent = x.getParent();
            Node sibling = this.getOtherChild(parent, x);
            Node grandparent = parent.getParent();

            // the branches of the pruned tree, given by their bottom node, with the top of each
            List<Node> branches = new ArrayList<>();
            Map<Node, Node> tops = new HashMap<>();
            for (Node node : tree.getNodesAsArray()) {
                if (node.isRoot() || node == parent || this.isInSubtree(node, x)) {
                    continue;
                }
                Node top = node == sibling ? grandparent : node.getParent();
                if (top.getHeight() > x.getHeight()) {
                    branches.add(node);
                    tops.put(node, top);
                }
            }

            Node newSibling = branches.get(Randomizer.nextInt(branches.size()));
            Node newGrandparent = tops.get(newSibling);
            double newLower = Math.max(newSibling.getHeight(), x.getHeight());
            double newUpper = newGrandparent.getHeight();
            double oldLower = Math.max(sibling.getHeight(), x.getHeight());
            double oldUpper = grandparent.getHeight();
            double newHeight = newLower + Randomizer.nextDouble() * (newUpper - newLower);

            NodeStates nodeStates = this.nodeStatesInput.get();
            if (newSibling != sibling) {
                this.replace(grandparent, parent, sibling);
                this.replace(parent, sibling, newSibling);
                this.replace(newGrandparent, newSibling, parent);
            }
            parent.setHeight(newHeight);
            if (newSibling != sibling) {
                nodeStates.markBranchChanged(parent, this);
                nodeStates.markBranchChanged(newSibling, this);
                nodeStates.markBranchChanged(sibling, this);
            }

            return Math.log(newUpper - newLower) - Math.log(oldUpper - oldLower);
        }

        private boolean isInSubtree(Node node, Node subtreeRoot) {
            for (Node ancestor = node; ancestor != null; ancestor = ancestor.getParent()) {
                if (ancestor == subtreeRoot) {
                    return true;
                }
            }
            return false;
        }

    }

}
