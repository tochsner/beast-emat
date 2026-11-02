package emat.adapters;

import beast.base.core.Description;

@Description("Adapter with a single mutable value, the logarithm of minus the sum of the logarithms of the ratios of the " +
        "internal nodes below the root to their parents. Changing it raises every ratio to the same power and " +
        "recomputes the heights from the root down, which keeps the root, the relative positions of the mutations " +
        "on their branches, the topology and the tips.")
public class RatioPowerAdapter extends RatioLineAdapter {

    // the logarithm of the power of the current state with respect to the ratios stored by refresh(), and the sum of the log ratios stored by refresh()
    private double logPower;
    private double baseSumLogRatios;

    @Override
    public void refresh() {
        super.refresh();

        this.logPower = 0.0;
        this.baseSumLogRatios = 0.0;
        for (int i = 1; i < this.internalNodes.length; i++) {
            this.baseSumLogRatios += Math.log(this.baseRatios[this.internalNodes[i].getNr()]);
        }
    }

    /** The coordinate is an absolute one, so that the states of different proposals can be compared: the log of minus the sum of the log ratios. */
    @Override
    public double[] getMutable(int nodeId) {
        return new double[]{this.logPower + Math.log(-this.baseSumLogRatios)};
    }

    /** Raises the ratios stored by refresh() to the power at which minus the sum of their logarithms has the given logarithm. */
    @Override
    public double update(double[] mutable, int nodeId) {
        this.logPower = mutable[0] - Math.log(-this.baseSumLogRatios);
        this.applyHeights(this.internalNodes[0].getHeight(), Math.exp(this.logPower));
        return 0.0;
    }

    /**
     * Returns minus the log of the volume element along the line. With the power k, the log ratios k L are k times
     * those of the state at the start, so with the coordinate log k a line through m ratios has the volume element
     * k^m Π r, as the ratios are the exponential of their logarithms. The Jacobian of the ratios to the heights and the
     * mutation times contribute their volume element at the current state. Slice sampling along the line then draws
     * from the posterior density of the line coordinate.
     */
    @Override
    public double getLogJacobianCorrection(int nodeId) {
        int numRatios = this.internalNodes.length - 1;
        double logVolume = numRatios * this.logPower + Math.exp(this.logPower) * this.baseSumLogRatios;
        return -(logVolume + this.computeLogRatioVolume());
    }

}
