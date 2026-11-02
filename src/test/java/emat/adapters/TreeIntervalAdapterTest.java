package emat.adapters;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import adapters.Adapter;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.operators.nodetimes.RatioScaleOperator;
import emat.operators.nodetimes.TiltedIntervalScaleOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;
import slice.LinCombSliceOperator;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TreeIntervalAdapterTest {

    private static final String[] SEQUENCES = {"ACG", "ACT", "CCT"};

    private static final int NUM_BURNIN = 20000;
    private static final int NUM_SAMPLES = 1000000;

    /**
     * On the three-tip tree ((A,B),C) with an exponential prior on both node heights, runs a chain that cycles
     * through slice sampling along the interval scaling line, which keeps the ratios of the intervals, the tilt
     * operator, which changes the ratios, and the site history Gibbs operator. The chain targets p(h1, h2) ∝ exp(-h1 - h2) L(h1, h2) for h1 < h2 with the
     * Jukes-Cantor likelihood L, whose means are compared with the sampled means.
     */
    @Test
    public void testThreeTipsMatchExactPosterior() {
        this.assertChainMatchesExactPosterior(false, Line.INTERVAL);
    }

    /**
     * The same chain with a clock rate with a Gamma(2, 1) prior, which the slice scales inversely to the tree length
     * and a scale move changes on its own. The exact posterior integrates the clock rate on a grid.
     */
    @Test
    public void testThreeTipsWithClockMatchExactPosterior() {
        this.assertChainMatchesExactPosterior(true, Line.INTERVAL);
    }

    /** The lines through the state that the slice samples along. */
    private enum Line {INTERVAL, ROOT_RATIO, POWER}

    private void assertChainMatchesExactPosterior(boolean hasClock, Line line) {
        Randomizer.setSeed(1);

        Alignment alignment = new Alignment();
        List<Sequence> sequences = List.of(new Sequence("A", SEQUENCES[0]), new Sequence("B", SEQUENCES[1]), new Sequence("C", SEQUENCES[2]));
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):0.5,C:1.5);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());
        RealScalarParam<PositiveReal> clockRate = new RealScalarParam<>(1.0, PositiveReal.INSTANCE);
        // the clock model refreshes its cached rate and dirtiness with the state, which this test does not have
        StrictClockModel clockModel = new StrictClockModel() {
            @Override
            public double getRateForBranch(Node node) {
                return clockRate.get();
            }

            @Override
            public boolean somethingIsDirty() {
                return true;
            }
        };
        clockModel.initByName("clock.rate", clockRate);
        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        Adapter adapter = this.createAdapter(line, tree, mutations, hasClock ? clockRate : null);
        LinCombSliceOperator sliceOperator = new LinCombSliceOperator();
        sliceOperator.initByName("weight", 1.0, "adapter", adapter);

        // the slice keeps the ratios of the intervals or the ratios to the parents, so another move has to change them
        Operator moveOperator;
        if (line == Line.INTERVAL) {
            TiltedIntervalScaleOperator tiltOperator = new TiltedIntervalScaleOperator();
            tiltOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
            tiltOperator.setOperatorSchedule(new OperatorSchedule());
            moveOperator = tiltOperator;
        } else {
            RatioScaleOperator ratioOperator = new RatioScaleOperator();
            ratioOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
            ratioOperator.setOperatorSchedule(new OperatorSchedule());
            moveOperator = ratioOperator;
        }
        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        Node root = tree.getRoot();
        Node cherry = root.getLeft().isLeaf() ? root.getRight() : root.getLeft();

        double sumCherryHeights = 0.0;
        double sumRootHeights = 0.0;
        int numSlices = 0;

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            // the Gibbs operator and the slice operator draw from exact conditionals, so they are always accepted

            gibbsOperator.proposal();

            if (hasClock) {
                this.runClockStep(tree, geneticPrior, clockRate);
            }

            if (i % 2 == 0) {
                // a slice proposal that fails leaves the state as it was
                double result = sliceOperator.proposal(() -> this.computeLogTarget(tree, geneticPrior, hasClock ? clockRate : null), null);
                numSlices += result != Double.NEGATIVE_INFINITY ? 1 : 0;
            } else {
                this.runMetropolisHastingsStep(tree, mutations, geneticPrior, moveOperator, hasClock ? clockRate : null);
            }

            if (i >= NUM_BURNIN) {
                sumCherryHeights += cherry.getHeight();
                sumRootHeights += root.getHeight();
            }
        }

        assertEquals(true, numSlices > 100000, "successful slices " + numSlices);

        double[] expectedHeights = this.computeExactMeanHeights(hasClock);
        assertEquals(expectedHeights[0], sumCherryHeights / NUM_SAMPLES, 0.02);
        assertEquals(expectedHeights[1], sumRootHeights / NUM_SAMPLES, 0.03);
    }

    /** Slice sampling along the line that scales the root above its oldest tip and keeps the ratios to the parents. */
    @Test
    public void testRootRatioLineMatchesExactPosterior() {
        this.assertChainMatchesExactPosterior(false, Line.ROOT_RATIO);
    }

    /** Slice sampling along the line that raises the ratios to the parents to a common power and keeps the root. */
    @Test
    public void testRatioPowerLineMatchesExactPosterior() {
        this.assertChainMatchesExactPosterior(false, Line.POWER);
    }

    private Adapter createAdapter(Line line, TreeParser tree, Mutations mutations, RealScalarParam<PositiveReal> clockRate) {
        switch (line) {
            case ROOT_RATIO -> {
                RootRatioAdapter adapter = new RootRatioAdapter();
                adapter.initByName("tree", tree, "mutations", mutations);
                return adapter;
            }
            case POWER -> {
                RatioPowerAdapter adapter = new RatioPowerAdapter();
                adapter.initByName("tree", tree, "mutations", mutations);
                return adapter;
            }
            default -> {
                TreeIntervalAdapter adapter = new TreeIntervalAdapter();
                if (clockRate != null) {
                    adapter.initByName("tree", tree, "mutations", mutations, "down", clockRate);
                } else {
                    adapter.initByName("tree", tree, "mutations", mutations);
                }
                return adapter;
            }
        }
    }

    /** Proposes with the given operator and accepts or restores the heights and mutations. Returns whether the proposal was accepted. */
    private boolean runMetropolisHastingsStep(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior, Operator operator, RealScalarParam<PositiveReal> clockRate) {
        Node[] nodes = tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        List<List<Mutation>> oldMutations = new ArrayList<>();
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
            oldMutations.add(mutations.getMutations(node));
        }
        double logTarget = this.computeLogTarget(tree, geneticPrior, clockRate);
        double oldRate = clockRate == null ? 0.0 : clockRate.get();

        double logHastingsRatio = operator.proposal();
        double newLogTarget = logHastingsRatio == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY : this.computeLogTarget(tree, geneticPrior, clockRate);

        if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
            operator.accept();
            return true;
        }

        operator.reject(logHastingsRatio == Double.NEGATIVE_INFINITY ? -2 : 0);
        if (clockRate != null) {
            clockRate.set(oldRate);
        }
        for (Node node : nodes) {
            node.setHeight(oldHeights[node.getNr()]);
        }
        for (Node node : nodes) {
            mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
        }
        return false;
    }

    /** Metropolis-Hastings step that scales the clock rate, whose Gamma(2, 1) prior is on the rate itself. */
    private void runClockStep(TreeParser tree, GeneticPrior geneticPrior, RealScalarParam<PositiveReal> clockRate) {
        double oldRate = clockRate.get();
        double logTarget = this.computeLogTarget(tree, geneticPrior, clockRate);
        double scaler = Math.exp(0.4 * (2.0 * Randomizer.nextDouble() - 1.0));

        clockRate.set(oldRate * scaler);
        double newLogTarget = this.computeLogTarget(tree, geneticPrior, clockRate);

        if (Math.log(Randomizer.nextDouble()) >= newLogTarget - logTarget + Math.log(scaler)) {
            clockRate.set(oldRate);
        }
    }

    private double computeLogTarget(TreeParser tree, GeneticPrior geneticPrior, RealScalarParam<PositiveReal> clockRate) {
        double sumInternalHeights = 0.0;
        for (Node node : tree.getInternalNodes()) {
            sumInternalHeights += node.getHeight();
        }
        double logClockPrior = clockRate == null ? 0.0 : Math.log(clockRate.get()) - clockRate.get();
        return -sumInternalHeights + logClockPrior + geneticPrior.calculateLogP();
    }

    /** Returns the exact posterior means of the heights of the cherry and the root, integrated by the midpoint rule on a grid. */
    private double[] computeExactMeanHeights(boolean hasClock) {
        if (hasClock) {
            return this.computeExactMeanHeightsWithClock();
        }

        int numSteps = 1500;
        double maxHeight = 15.0;
        double stepSize = maxHeight / numSteps;

        double normalisation = 0.0;
        double meanCherryHeight = 0.0;
        double meanRootHeight = 0.0;

        for (int i = 0; i < numSteps; i++) {
            double cherryHeight = (i + 0.5) * stepSize;
            for (int j = i; j < numSteps; j++) {
                double rootHeight = (j + 0.5) * stepSize;
                double weight = (j == i ? 0.5 : 1.0) * Math.exp(-cherryHeight - rootHeight) * this.computeLikelihood(cherryHeight, rootHeight);

                normalisation += weight;
                meanCherryHeight += weight * cherryHeight;
                meanRootHeight += weight * rootHeight;
            }
        }

        return new double[]{meanCherryHeight / normalisation, meanRootHeight / normalisation};
    }

    /** Returns the exact posterior means of the heights with a Gamma(2, 1) clock rate that is integrated out on a grid. */
    private double[] computeExactMeanHeightsWithClock() {
        int numSteps = 800;
        double maxHeight = 15.0;
        double stepSize = maxHeight / numSteps;
        int numRates = 300;
        double maxRate = 10.0;

        double normalisation = 0.0;
        double meanCherryHeight = 0.0;
        double meanRootHeight = 0.0;

        for (int k = 0; k < numRates; k++) {
            double rate = (k + 0.5) * maxRate / numRates;
            for (int i = 0; i < numSteps; i++) {
                double cherryHeight = (i + 0.5) * stepSize;
                for (int j = i; j < numSteps; j++) {
                    double rootHeight = (j + 0.5) * stepSize;
                    double weight = (j == i ? 0.5 : 1.0) * rate * Math.exp(-rate - cherryHeight - rootHeight)
                            * this.computeLikelihood(rate * cherryHeight, rate * rootHeight);

                    normalisation += weight;
                    meanCherryHeight += weight * cherryHeight;
                    meanRootHeight += weight * rootHeight;
                }
            }
        }

        return new double[]{meanCherryHeight / normalisation, meanRootHeight / normalisation};
    }

    /** Returns the Jukes-Cantor likelihood of the three-tip tree ((A,B),C) with the given heights of the cherry and the root. */
    private double computeLikelihood(double cherryHeight, double rootHeight) {
        double likelihood = 1.0;

        for (int site = 0; site < SEQUENCES[0].length(); site++) {
            char a = SEQUENCES[0].charAt(site);
            char b = SEQUENCES[1].charAt(site);
            char c = SEQUENCES[2].charAt(site);

            double siteLikelihood = 0.0;
            for (char rootState : "ACGT".toCharArray()) {
                for (char cherryState : "ACGT".toCharArray()) {
                    siteLikelihood += 0.25
                            * this.getJukesCantorProbability(rootState == cherryState, rootHeight - cherryHeight)
                            * this.getJukesCantorProbability(cherryState == a, cherryHeight)
                            * this.getJukesCantorProbability(cherryState == b, cherryHeight)
                            * this.getJukesCantorProbability(rootState == c, rootHeight);
                }
            }
            likelihood *= siteLikelihood;
        }

        return likelihood;
    }

    /** Returns the Jukes-Cantor probability of ending in the same or in a given other state over the given time. */
    private double getJukesCantorProbability(boolean isSame, double time) {
        double decay = Math.exp(-4.0 * time / 3.0);
        return isSame ? 0.25 + 0.75 * decay : 0.25 - 0.25 * decay;
    }

}
