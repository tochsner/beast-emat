package emat.operators.treeprior;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LearnedTreePriorOperatorTest {

    private static final double LOG_MEAN = 2.0;
    private static final double LOG_SD = 0.3;

    /** Moves the logarithm of a positive parameter. */
    private static class LogAdapter extends BEASTObject implements Adapter {

        private final RealScalarParam<PositiveReal> parameter;

        private LogAdapter(RealScalarParam<PositiveReal> parameter) {
            this.parameter = parameter;
        }

        @Override
        public void initAndValidate() {
        }

        @Override
        public int getNumImmutable() {
            return 0;
        }

        @Override
        public int getNumMutable() {
            return 1;
        }

        @Override
        public double[] getImmutable(int nodeId) {
            return new double[0];
        }

        @Override
        public double[] getMutable(int nodeId) {
            return new double[]{Math.log(this.parameter.get())};
        }

        @Override
        public double update(double[] mutable, int nodeId) {
            this.parameter.set(Math.exp(mutable[0]));
            return 0.0;
        }

        @Override
        public double getLogJacobianCorrection(int nodeId) {
            return -Math.log(this.parameter.get());
        }

        @Override
        public List<StateNode> listStateNodes() {
            return List.of(this.parameter);
        }

    }

    /** A log-normal density of the parameter. */
    private static class LogNormalDensity extends Distribution {

        private final RealScalarParam<PositiveReal> parameter;

        private LogNormalDensity(RealScalarParam<PositiveReal> parameter) {
            this.parameter = parameter;
        }

        @Override
        public void initAndValidate() {
        }

        @Override
        public double calculateLogP() {
            double x = Math.log(this.parameter.get());
            this.logP = -0.5 * Math.pow((x - LOG_MEAN) / LOG_SD, 2) - x;
            return this.logP;
        }

        @Override
        public List<String> getArguments() {
            return List.of();
        }

        @Override
        public List<String> getConditions() {
            return List.of();
        }

        @Override
        public void sample(beast.base.inference.State state, Random random) {
        }

    }

    @Test
    public void acceptsAlmostEveryProposalWhenTheTargetIsGaussian() {
        Randomizer.setSeed(7);
        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((a:1.0,b:1.0):1.0,(c:0.25,d:0.5):1.5)", "IsLabelledNewick", true, "adjustTipHeights", false);
        RealScalarParam<PositiveReal> parameter = new RealScalarParam<>(5.0, PositiveReal.INSTANCE);
        State state = new State();
        state.initByName("stateNode", parameter);
        LogNormalDensity density = new LogNormalDensity(parameter);
        state.initialise();
        state.setPosterior(density);
        LogAdapter adapter = new LogAdapter(parameter);

        LearnedTreePriorOperator operator = new LearnedTreePriorOperator();
        operator.initByName("weight", 1.0, "tree", tree, "state", state, "adapter", adapter, "distribution", density,
                "sweepEvery", 1, "numDraws", 4, "minRecords", 200, "burnIn", 10);

        int numAccepted = 0;
        int numProposed = 0;
        double sum = 0.0;
        for (int i = 0; i < 3000; i++) {
            double oldLogTarget = this.computeLogTarget(parameter);
            double oldValue = parameter.get();
            double logHastings = operator.proposal();
            if (i < 1000) {
                continue;
            }

            numProposed++;
            double logAlpha = this.computeLogTarget(parameter) - oldLogTarget + logHastings;
            if (logHastings > Double.NEGATIVE_INFINITY && Math.log(Randomizer.nextDouble()) < logAlpha) {
                numAccepted++;
            } else {
                parameter.set(oldValue);
            }
            sum += Math.log(parameter.get());
        }

        assertTrue(numAccepted > 0.8 * numProposed, "acceptance " + numAccepted / (double) numProposed);
        assertEquals(LOG_MEAN, sum / numProposed, 0.05);
    }

    /** Returns the log density of the parameter, which has the log-normal density of its logarithm divided by the parameter. */
    private double computeLogTarget(RealScalarParam<PositiveReal> parameter) {
        double x = Math.log(parameter.get());
        return -0.5 * Math.pow((x - LOG_MEAN) / LOG_SD, 2) - x;
    }

}
