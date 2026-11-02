package emat.adapters;

import beast.base.core.Description;

@Description("Adapter with a single mutable value, the logarithm of the height of the root above its oldest tip. " +
        "Changing it moves the root and recomputes every other internal height from its unchanged ratio to its " +
        "parent, which keeps the relative positions of the mutations on their branches, the topology and the tips.")
public class RootRatioAdapter extends RatioLineAdapter {

    @Override
    public double[] getMutable(int nodeId) {
        return new double[]{this.computeCurrentLogRootHeight()};
    }

    /** Moves the root to the given logarithm of its height above its oldest tip and recomputes the heights below from the ratios stored by refresh(). */
    @Override
    public double update(double[] mutable, int nodeId) {
        double rootHeight = this.oldestTipHeights[this.internalNodes[0].getNr()] + Math.exp(mutable[0]);
        this.applyHeights(rootHeight, 1.0);
        return 0.0;
    }

    /**
     * Returns minus the log of the volume element along the line: the coordinate is the logarithm of the height x
     * of the root above its oldest tip, which has the volume element x, and the ratios and the mutation times
     * contribute their volume element at the current state. Slice sampling along the line then draws from the
     * posterior density of the line coordinate.
     */
    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return -(this.computeCurrentLogRootHeight() + this.computeLogRatioVolume());
    }

    private double computeCurrentLogRootHeight() {
        return Math.log(this.tree.getRoot().getHeight() - this.oldestTipHeights[this.internalNodes[0].getNr()]);
    }

}
