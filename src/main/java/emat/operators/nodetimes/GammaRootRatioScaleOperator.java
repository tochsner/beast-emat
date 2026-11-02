package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;

@Description("Moves the root like the root ratio scale operator, but draws the factor s by which the height of the " +
        "root above its oldest tip is scaled from a Gamma distribution with shape k and rate k, so that s has mean one. " +
        "The kernel is skewed in log s, with a long tail towards small factors and a sharp limit towards large ones, " +
        "like the posterior along the line of the move. The width 1 / √k of the kernel is tuned towards the target acceptance.")
public class GammaRootRatioScaleOperator extends RootRatioScaleOperator {

    final public Input<Double> targetAcceptanceInput = new Input<>("targetAcceptance", "acceptance probability towards which the width of the kernel is tuned", 0.5);

    private double targetAcceptance;

    // the shape k of the kernel of the last proposal
    private double shape;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.targetAcceptance = this.targetAcceptanceInput.get();
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.targetAcceptance;
    }

    /** Draws s for the root from Gamma(k, k) with k = 1 / w² for the given width w, and for any other node from the default kernel. */
    @Override
    protected double drawScaler(Node node, double width) {
        if (!node.isRoot()) {
            return super.drawScaler(node, width);
        }

        this.shape = 1.0 / (width * width);
        return Randomizer.nextGamma(this.shape, this.shape);
    }

    /**
     * Returns log f(1 / s) - log f(s) - log s for the density f of Gamma(k, k), which is the ratio of the reverse to the
     * forward proposal density of the height x of the root above its oldest tip, as x' = s x has the density f(s) / x
     * and the reverse move the density f(1 / s) / x'. The normalising constants of the density cancel.
     */
    @Override
    protected double computeRootLogHastingsRatio(double scaler) {
        return -(2.0 * this.shape - 1.0) * Math.log(scaler) - this.shape * (1.0 / scaler - scaler);
    }

}
