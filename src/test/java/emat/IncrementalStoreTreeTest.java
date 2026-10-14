package emat;

import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.Exchange;
import beast.base.evolution.operator.ScaleOperator;
import beast.base.evolution.operator.SubtreeSlide;
import beast.base.evolution.operator.Uniform;
import beast.base.evolution.operator.WilsonBalding;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import emat.tree.IncrementalStoreNode;
import emat.tree.IncrementalStoreTree;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

public class IncrementalStoreTreeTest {

    private static final int NUM_TAXA = 30;
    private static final int NUM_STEPS = 20000;
    private static final Duration TIMEOUT = Duration.ofSeconds(60);

    /**
     * Runs the same chain of core tree operators with random acceptances on a core tree and on
     * an incremental store tree, and checks that both trees pass through the same states.
     */
    @Test
    public void testMatchesCoreTree() {
        // both chains run on the same thread, as the randomizer seeds the threads differently
        assertTimeoutPreemptively(TIMEOUT, () -> {
            List<String> expectedStates = this.runChain(11, new Tree(), false);
            List<String> actualStates = this.runChain(11, new IncrementalStoreTree(), false);

            assertEquals(expectedStates.size(), actualStates.size());
            for (int i = 0; i < expectedStates.size(); i++) {
                assertEquals(expectedStates.get(i), actualStates.get(i), "states differ at step " + i);
            }
        });
    }

    /**
     * Checks that every rejected proposal restores the tree to the state before the proposal,
     * including the dirtiness flags when restoring without cleaning the state afterwards.
     */
    @Test
    public void testRestoresPreviousState() {
        assertTimeoutPreemptively(TIMEOUT, () -> this.runChain(13, new IncrementalStoreTree(), true));
    }

    @Test
    public void testConvertsParsedNodes() {
        TreeParser parser = new TreeParser("((A:1,B:1):1,(C:0.5,D:0.5):1.5);");

        IncrementalStoreTree tree = new IncrementalStoreTree();
        tree.setRoot(parser.getRoot());
        tree.initArrays();

        for (Node node : tree.getNodesAsArray()) {
            assertInstanceOf(IncrementalStoreNode.class, node);
            assertEquals(tree, node.getTree());
        }
        assertEquals(parser.getRoot().toNewick(), tree.getRoot().toNewick());
    }

    private List<String> runChain(long seed, Tree tree, boolean isCheckingRestores) {
        Randomizer.setSeed(seed);

        List<Taxon> taxa = new ArrayList<>();
        for (int i = 0; i < NUM_TAXA; i++) {
            taxa.add(new Taxon("t" + i));
        }
        tree.initByName("taxonset", new TaxonSet(taxa));

        State state = new State();
        state.initByName("stateNode", tree);
        state.initialise();
        state.setEverythingDirty(true);
        state.setEverythingDirty(false);

        List<Operator> operators = this.createOperators(tree);
        List<String> states = new ArrayList<>();

        for (long i = 0; i < NUM_STEPS; i++) {
            String previousState = this.describe(tree);

            state.store(i);
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));
            double logHastingsRatio = operator.proposal();

            if (logHastingsRatio == Double.NEGATIVE_INFINITY || Randomizer.nextDouble() < 0.5) {
                state.restore();
                if (isCheckingRestores) {
                    assertEquals(previousState, this.describe(tree), "restore differs at step " + i);
                }
            }
            state.setEverythingDirty(false);

            String currentState = this.describe(tree);
            if (isCheckingRestores) {
                this.assertConsistent(tree);
            }
            states.add(currentState);
        }
        return states;
    }

    private List<Operator> createOperators(Tree tree) {
        SubtreeSlide subtreeSlide = new SubtreeSlide();
        subtreeSlide.initByName("tree", tree, "weight", 1.0, "size", 2.0, "optimise", false);

        Exchange narrowExchange = new Exchange();
        narrowExchange.initByName("tree", tree, "weight", 1.0, "isNarrow", true);

        Exchange wideExchange = new Exchange();
        wideExchange.initByName("tree", tree, "weight", 1.0, "isNarrow", false);

        WilsonBalding wilsonBalding = new WilsonBalding();
        wilsonBalding.initByName("tree", tree, "weight", 1.0);

        Uniform uniform = new Uniform();
        uniform.initByName("tree", tree, "weight", 1.0);

        ScaleOperator scaleOperator = new ScaleOperator();
        scaleOperator.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9, "optimise", false);

        ScaleOperator rootScaleOperator = new ScaleOperator();
        rootScaleOperator.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9, "rootOnly", true, "optimise", false);

        return List.of(subtreeSlide, narrowExchange, wideExchange, wilsonBalding, uniform, scaleOperator, rootScaleOperator);
    }

    /**
     * Describes the nodes in the order of their numbers, including heights, parents, children
     * and dirtiness flags.
     */
    private String describe(Tree tree) {
        StringBuilder description = new StringBuilder("root=" + tree.getRoot().getNr() + "\n");
        for (Node node : tree.getNodesAsArray()) {
            description.append(node.getNr())
                    .append(" h=").append(node.getHeight())
                    .append(" p=").append(node.isRoot() ? -1 : node.getParent().getNr())
                    .append(" d=").append(node.isDirty())
                    .append(" c=");
            for (Node child : node.getChildren()) {
                description.append(child.getNr()).append(",");
            }
            description.append("\n");
        }
        return description.toString();
    }

    private void assertConsistent(Tree tree) {
        Node[] nodes = tree.getNodesAsArray();
        assertEquals(nodes.length, tree.getRoot().getNodeCount());
        for (int i = 0; i < nodes.length; i++) {
            assertEquals(i, nodes[i].getNr());
            assertEquals(Tree.IS_CLEAN, nodes[i].isDirty());
            for (Node child : nodes[i].getChildren()) {
                assertEquals(nodes[i], child.getParent());
                assertEquals(nodes[child.getNr()], child);
            }
        }
    }
}
