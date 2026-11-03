package emat.operators.nodetimes;

import beast.base.core.Input;
import beast.base.inference.Scalable;

import java.util.ArrayList;
import java.util.List;

/**
 * A scale operator that can scale other parameters along with the tree length, so that the product of the tree length and
 * a parameter such as the clock rate stays the same.
 */
public abstract class CompensatedScaleOperator extends ScaleTunedOperator {

    final public Input<List<Scalable>> downInput = new Input<>("down", "parameters to scale inversely to the tree length", new ArrayList<>());
    final public Input<List<Scalable>> upInput = new Input<>("up", "parameters to scale with the tree length", new ArrayList<>());

    /** Returns whether any parameter is scaled along with the tree length. */
    protected boolean hasCompensatedParameters() {
        return !this.downInput.get().isEmpty() || !this.upInput.get().isEmpty();
    }

    /**
     * Scales the parameters with the factor by which the tree length changed, and returns the log Jacobian of that scaling,
     * or negative infinity if a parameter rejects its new value. The factor is a function of the old and the new tree, so the
     * reverse move applies the inverse factor and the parameters take part in the bijection.
     */
    protected double scaleCompensatedParameters(double lengthBefore, double lengthAfter) {
        double actualScaler = lengthAfter / lengthBefore;
        double logJacobian = 0.0;

        // scale returns the log Jacobian, i.e. dimension * log(factor)

        try {
            for (Scalable down : this.downInput.get()) {
                logJacobian += down.scale(1.0 / actualScaler);
            }
            for (Scalable up : this.upInput.get()) {
                logJacobian += up.scale(actualScaler);
            }
        } catch (IllegalArgumentException e) {
            // a scalable rejected the move
            return Double.NEGATIVE_INFINITY;
        }

        return logJacobian;
    }

}
