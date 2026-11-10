package esat.stochasticmapping;

import beast.base.util.Randomizer;

/** Draws from categorical distributions given by unnormalised weights. */
public final class Categorical {

    private Categorical() {
    }

    /**
     * Draws a state with probability proportional to its weight, where the weight of state x
     * is weights[offset + x] and the weights sum to the given total. A rounding error in the
     * total falls back to the last state with a positive weight.
     */
    public static int drawState(double[] weights, int offset, int numStates, double total) {
        if (!(total > 0.0) || !Double.isFinite(total)) {
            throw new IllegalStateException("Cannot draw a state from weights with total " + total + ".");
        }

        double remaining = Randomizer.nextDouble() * total;
        int lastPositiveState = -1;
        for (int state = 0; state < numStates; state++) {
            double weight = weights[offset + state];
            if (weight > 0.0) {
                lastPositiveState = state;
                remaining -= weight;
                if (remaining < 0.0) {
                    return state;
                }
            }
        }
        return lastPositiveState;
    }

}
