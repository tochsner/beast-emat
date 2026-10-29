package emat.operators.nodetimes;

import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.junit.jupiter.api.Test;

/** Checks that every reparameterisation operator samples the exact posterior of the fixed-history test of the ratio operator. */
public class ReparamScaleOperatorTest {

    private final RatioScaleOperatorTest exactPosteriorTest = new RatioScaleOperatorTest();

    @Test
    public void testTipOffsetMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new TipOffsetScaleOperator());
    }

    @Test
    public void testGapShiftMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new GapShiftScaleOperator());
    }

    @Test
    public void testPresentRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new PresentRatioScaleOperator());
    }

    @Test
    public void testYoungestTipRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new YoungestTipRatioScaleOperator());
    }

    @Test
    public void testRootAnchoredRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new RootAnchoredRatioScaleOperator());
    }

    @Test
    public void testTunedRootAnchoredRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new RootAnchoredRatioScaleOperator(), "optimise", true);
    }

    @Test
    public void testGammaTransportRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new GammaTransportRatioScaleOperator(), "minObservations", 50, "refitInterval", 50);
    }

    @Test
    public void testTunedGammaTransportRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new GammaTransportRatioScaleOperator(), "minObservations", 50, "refitInterval", 50, "optimise", true);
    }

    @Test
    public void testGammaRegressionRatioMatchesExactPosterior() {
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new GammaRegressionRatioScaleOperator(), "minObservations", 50);
    }

    @Test
    public void testTunedGammaRegressionRatioWithTreePriorParameterMatchesExactPosterior() {
        RealScalarParam<PositiveReal> treePriorParameter = new RealScalarParam<>(0.3, PositiveReal.INSTANCE);
        this.exactPosteriorTest.assertFixedHistoryMatchesExactPosterior(new GammaRegressionRatioScaleOperator(),
                "minObservations", 50, "treePriorParameter", treePriorParameter, "optimise", true);
    }

}
