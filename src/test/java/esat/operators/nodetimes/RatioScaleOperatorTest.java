package esat.operators.nodetimes;

import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import esat.EsatFixture;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RatioScaleOperatorTest {

    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGT"),
            new Sequence("B", "ACGA"),
            new Sequence("C", "CCGA"),
            new Sequence("D", "CTTA")
    );

    // a caterpillar (((A,B),C),D) whose tips C and D are sampled before A and B
    private static final String NEWICK = "(((A:0.5,B:0.5):0.5,C:0.7):0.5,D:1.3);";

    private static final int NUM_BURNIN = 10000;
    private static final int NUM_SAMPLES = 400000;

    // the midpoint rule on a grid of n points has an error linear in 1/n, which extrapolating from n and 2n removes
    private static final int NUM_GRID_POINTS = 300;
    private static final double MAX_GRID_HEIGHT = 8.0;

    @Test
    public void testFixedStatesMatchExactPosterior() {
        this.assertFixedStatesMatchExactPosterior(false);
    }

    /**
     * The step of a proposal depends only on the size of the picked clade, which the move
     * keeps, so tuning the step of every clade size class still targets the exact posterior.
     */
    @Test
    public void testTunedFixedStatesMatchExactPosterior() {
        this.assertFixedStatesMatchExactPosterior(true);
    }

    /**
     * The root operator alone only moves the root and the heights in proportion to it, so it
     * is mixed with the operator on all internal nodes, which makes one proposal in five.
     */
    @Test
    public void testRootMixedWithRatioMatchesExactPosterior() {
        this.assertFixedStatesMatchExactPosterior(true, true);
    }

    /**
     * The batch operator draws from an approximation of the genetic prior, which is either its tangent at the midpoint
     * of each range or a piecewise exponential interpolation, and corrects for the deviation, so it keeps the exact
     * posterior, with and without the learned approximation of the tree prior. The root moves with the root operator.
     */
    @Test
    public void testTangentBatchNodeDisplacementMatchesExactPosterior() {
        this.assertBatchMatchesExactPosterior(false, false);
    }

    @Test
    public void testLearnedTangentBatchNodeDisplacementMatchesExactPosterior() {
        this.assertBatchMatchesExactPosterior(true, false);
    }

    @Test
    public void testInterpolatedBatchNodeDisplacementMatchesExactPosterior() {
        this.assertBatchMatchesExactPosterior(false, true);
    }

    @Test
    public void testLearnedInterpolatedBatchNodeDisplacementMatchesExactPosterior() {
        this.assertBatchMatchesExactPosterior(true, true);
    }

    /** The root scale operator moves the root only, so the inner nodes move with the batch operator. */
    @Test
    public void testRootScaleWithBatchNodeDisplacementMatchesExactPosterior() {
        this.assertBatchMatchesExactPosterior(false, true, true);
    }

    private void assertBatchMatchesExactPosterior(boolean learnTreePrior, boolean interpolate) {
        this.assertBatchMatchesExactPosterior(learnTreePrior, interpolate, false);
    }

    private void assertBatchMatchesExactPosterior(boolean learnTreePrior, boolean interpolate, boolean useRootScale) {
        Randomizer.setSeed(34);
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler.Sample sample = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess).sample();
        fixture.nodeStates().initialiseStates(sample.rootSequence(), List.of(sample.changesAtNode()));

        double[] coarseMeans = this.integrateMeanHeights(fixture, substitutionProcess, NUM_GRID_POINTS);
        double[] fineMeans = this.integrateMeanHeights(fixture, substitutionProcess, 2 * NUM_GRID_POINTS);

        BatchNodeDisplacementOperator batchOperator = new BatchNodeDisplacementOperator();
        batchOperator.initByName("tree", fixture.tree(), "geneticPrior", fixture.geneticPrior(), "batchSize", 2, "optimise", false,
                "learnTreePrior", learnTreePrior, "interpolateGeneticPrior", interpolate, "treePriorWarmUp", 200, "weight", 1.0);
        TreeOperator rootOperator = useRootScale ? new RootScaleOperator() : new RootRatioScaleOperator();
        rootOperator.initByName("tree", fixture.tree(), "optimise", false, "scaleFactor", 0.5, "weight", 1.0);

        double[] sampledMeans = this.runChain(fixture, new TreeOperator[]{rootOperator, rootOperator, batchOperator, batchOperator});

        for (int i = 0; i < sampledMeans.length; i++) {
            assertEquals(2.0 * fineMeans[i] - coarseMeans[i], sampledMeans[i], 0.01);
        }
    }

    private void assertFixedStatesMatchExactPosterior(boolean optimise) {
        this.assertFixedStatesMatchExactPosterior(optimise, false);
    }

    /**
     * Runs the operator on the genetic prior with fixed states and the prior exp(-h) on the
     * root height, and compares the mean heights of the internal nodes with an integral over
     * a grid, extrapolated from two grid sizes.
     */
    private void assertFixedStatesMatchExactPosterior(boolean optimise, boolean withRootOperator) {
        Randomizer.setSeed(21);
        EsatFixture fixture = EsatFixture.create(NEWICK, SEQUENCES);
        SubstitutionProcess substitutionProcess = new SubstitutionProcess(fixture.geneticPrior());
        substitutionProcess.update();
        GlobalStateSampler.Sample sample = new GlobalStateSampler(fixture.nodeStates(), substitutionProcess).sample();
        fixture.nodeStates().initialiseStates(sample.rootSequence(), List.of(sample.changesAtNode()));

        double[] coarseMeans = this.integrateMeanHeights(fixture, substitutionProcess, NUM_GRID_POINTS);
        double[] fineMeans = this.integrateMeanHeights(fixture, substitutionProcess, 2 * NUM_GRID_POINTS);
        double[] expectedMeans = new double[coarseMeans.length];
        for (int i = 0; i < expectedMeans.length; i++) {
            expectedMeans[i] = 2.0 * fineMeans[i] - coarseMeans[i];
        }

        // a small fixed kernel mixes slowly on this posterior, so the untuned chain starts wider
        RatioScaleOperator operator = new RatioScaleOperator();
        operator.initByName("tree", fixture.tree(), "optimise", optimise, "scaleFactor", 0.5, "weight", 1.0);
        TreeOperator[] operators = {operator};
        if (withRootOperator) {
            RootRatioScaleOperator rootOperator = new RootRatioScaleOperator();
            rootOperator.initByName("tree", fixture.tree(), "optimise", optimise, "scaleFactor", 0.5, "weight", 1.0);
            operators = new TreeOperator[]{rootOperator, rootOperator, rootOperator, rootOperator, operator};
        }
        double[] sampledMeans = this.runChain(fixture, operators);

        for (int i = 0; i < expectedMeans.length; i++) {
            assertEquals(expectedMeans[i], sampledMeans[i], 0.01);
        }
    }

    /**
     * Integrates the means of the internal node heights of the caterpillar, from the cherry
     * to the root, by the midpoint rule on a regular grid of the given number of points. The log probability of each branch
     * only depends on its length, so it is tabulated once per branch over the grid.
     */
    private double[] integrateMeanHeights(EsatFixture fixture, SubstitutionProcess substitutionProcess, int numGridPoints) {
        Node[] internalNodes = this.getInternalNodes(fixture.tree());
        Node cherry = internalNodes[0];
        Node middle = internalNodes[1];
        double stepSize = MAX_GRID_HEIGHT / numGridPoints;

        // the branches above the tips end at a fixed height, and those between internal nodes span a whole number of steps

        double[][] tipLogPs = new double[fixture.tree().getNodeCount()][numGridPoints];
        for (Node tip : fixture.tree().getExternalNodes()) {
            for (int i = 0; i < numGridPoints; i++) {
                double length = (i + 0.5) * stepSize - tip.getHeight();
                tipLogPs[tip.getNr()][i] = this.computeBranchLogP(fixture, substitutionProcess, tip.getParent(), tip, length);
            }
        }

        double[] cherryLogPs = new double[numGridPoints];
        double[] middleLogPs = new double[numGridPoints];
        for (int k = 1; k < numGridPoints; k++) {
            cherryLogPs[k] = this.computeBranchLogP(fixture, substitutionProcess, middle, cherry, k * stepSize);
            middleLogPs[k] = this.computeBranchLogP(fixture, substitutionProcess, internalNodes[2], middle, k * stepSize);
        }

        Node[] tips = fixture.tree().getExternalNodes().toArray(new Node[0]);
        double normalisation = 0.0;
        double[] means = new double[internalNodes.length];
        for (int i = 0; i < numGridPoints; i++) {
            for (int j = i + 1; j < numGridPoints; j++) {
                for (int k = j + 1; k < numGridPoints; k++) {
                    int[] indices = {i, j, k};
                    double logDensity = -(k + 0.5) * stepSize + cherryLogPs[j - i] + middleLogPs[k - j];

                    boolean isValid = true;
                    for (Node tip : tips) {
                        int parentIndex = indices[this.indexOf(internalNodes, tip.getParent())];
                        if ((parentIndex + 0.5) * stepSize <= tip.getHeight()) {
                            isValid = false;
                            break;
                        }
                        logDensity += tipLogPs[tip.getNr()][parentIndex];
                    }
                    if (!isValid) {
                        continue;
                    }

                    double density = Math.exp(logDensity);
                    normalisation += density;
                    for (int n = 0; n < internalNodes.length; n++) {
                        means[n] += density * (indices[n] + 0.5) * stepSize;
                    }
                }
            }
        }

        for (int n = 0; n < internalNodes.length; n++) {
            means[n] /= normalisation;
        }
        return means;
    }

    /** Computes Σ N_ab log P_ab between the states of the given nodes on a branch of the given length at the rate of the lower node. */
    private double computeBranchLogP(EsatFixture fixture, SubstitutionProcess substitutionProcess, Node upper, Node lower, double length) {
        int[] counts = new int[EsatFixture.NUM_STATES * EsatFixture.NUM_STATES];
        fixture.nodeStates().countTransitions(upper, lower, counts);
        double[] probabilities = new double[counts.length];
        substitutionProcess.getTransitionProbabilities(lower, length, probabilities);

        double logP = 0.0;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] != 0) {
                logP += counts[i] * Math.log(probabilities[i]);
            }
        }
        return logP;
    }

    /** Returns the cherry (A, B), the middle node and the root of the caterpillar. */
    private Node[] getInternalNodes(Tree tree) {
        Node cherry = tree.getNode(tree.getTaxonset().getTaxonIndex("A")).getParent();
        return new Node[]{cherry, cherry.getParent(), tree.getRoot()};
    }

    private int indexOf(Node[] nodes, Node node) {
        for (int i = 0; i < nodes.length; i++) {
            if (nodes[i] == node) {
                return i;
            }
        }
        throw new IllegalArgumentException("Node is not internal.");
    }

    /**
     * Runs a Metropolis-Hastings chain against the genetic prior and the prior exp(-h) on the
     * root height h, whose every step uses one of the given operators drawn uniformly, and
     * returns the mean heights of the cherry, the middle node and the root after the burn-in.
     */
    private double[] runChain(EsatFixture fixture, TreeOperator[] operators) {
        Tree tree = fixture.tree();
        State state = new State();
        state.initByName("stateNode", tree, "stateNode", fixture.nodeStates());
        state.initialise();
        state.setPosterior(fixture.geneticPrior());
        state.setEverythingDirty(true);
        double logTarget = -tree.getRoot().getHeight() + fixture.geneticPrior().calculateLogP();
        state.setEverythingDirty(false);

        double[] sumHeights = new double[3];
        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            state.store(i);
            TreeOperator operator = operators[Randomizer.nextInt(operators.length)];
            double logHastingsRatio = operator.proposal();

            boolean isAccepted = false;
            double logAlpha = Double.NEGATIVE_INFINITY;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
                double newLogTarget = -tree.getRoot().getHeight() + fixture.geneticPrior().calculateLogP();
                logAlpha = newLogTarget - logTarget + logHastingsRatio;
                if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                    state.acceptCalculationNodes();
                    logTarget = newLogTarget;
                    isAccepted = true;
                }
            }

            // like the MCMC, the operator tunes itself after every proposal, which does nothing unless optimise is set
            operator.optimize(logAlpha);

            if (!isAccepted) {
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);

            if (i >= NUM_BURNIN) {
                Node[] internalNodes = this.getInternalNodes(tree);
                for (int n = 0; n < internalNodes.length; n++) {
                    sumHeights[n] += internalNodes[n].getHeight();
                }
            }
        }

        for (int n = 0; n < sumHeights.length; n++) {
            sumHeights[n] /= NUM_SAMPLES;
        }
        return sumHeights;
    }

}
