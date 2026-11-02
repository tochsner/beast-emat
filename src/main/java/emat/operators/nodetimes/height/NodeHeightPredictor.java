package emat.operators.nodetimes.height;

import java.util.List;

/**
 * Predicts the conditional log density of the height of an internal node between its parent and its older child, up to
 * an additive constant, by a quadratic ridge regression on {@link NodeHeightFeatures}. The densities are only known up
 * to a constant per node, so the regression learns from log densities that are centred within each batch of heights.
 */
public class NodeHeightPredictor {

    /** The log densities of one node at a batch of heights, which only need to be correct up to an additive constant. */
    public record Batch(NodeHeightContext context, double[] heights, double[] logDensities) {
    }

    private static final int NUM_FOLDS = 5;

    private final QuadraticRidgeRegression regression;

    /** Creates a predictor that chooses the penalty of the regression from the given grid. */
    public NodeHeightPredictor(double[] penalties) {
        this.regression = new QuadraticRidgeRegression(NodeHeightFeatures.COUNT, penalties);
    }

    public NodeHeightPredictor() {
        this(QuadraticRidgeRegression.createDefaultPenalties());
    }

    /**
     * Fits the predictor. The cross-validation that chooses the penalty holds out contiguous blocks of the batches, so
     * batches that are close in the list, such as those of the same state, are held out together.
     */
    public void fit(List<Batch> batches) {
        int numRows = batches.stream().mapToInt(batch -> batch.heights().length).sum();
        double[][] inputs = new double[numRows][];
        double[] targets = new double[numRows];
        int[] folds = new int[numRows];

        int row = 0;
        for (int batchNr = 0; batchNr < batches.size(); batchNr++) {
            Batch batch = batches.get(batchNr);
            double[] centered = NodeHeightPredictor.center(batch.logDensities());
            for (int i = 0; i < batch.heights().length; i++) {
                inputs[row] = NodeHeightFeatures.compute(batch.context(), batch.heights()[i]);
                targets[row] = centered[i];
                folds[row] = batchNr * NUM_FOLDS / batches.size();
                row++;
            }
        }

        this.regression.fit(inputs, targets, folds, NUM_FOLDS);
    }

    /** Returns the predicted log densities at the given heights, centred to a mean of zero. */
    public double[] predictLogDensities(NodeHeightContext context, double[] heights) {
        double[] predictions = new double[heights.length];
        for (int i = 0; i < heights.length; i++) {
            predictions[i] = this.regression.predict(NodeHeightFeatures.compute(context, heights[i]));
        }
        return NodeHeightPredictor.center(predictions);
    }

    /** Returns the predicted densities at the given heights, normalised to sum to one. */
    public double[] predictWeights(NodeHeightContext context, double[] heights) {
        double[] weights = this.predictLogDensities(context, heights);
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(weights[i]);
            sum += weights[i];
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= sum;
        }
        return weights;
    }

    /** Returns the penalty that the last fit chose. */
    public double getPenalty() {
        return this.regression.getPenalty();
    }

    private static double[] center(double[] values) {
        double mean = 0.0;
        for (double value : values) {
            mean += value / values.length;
        }
        double[] centered = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            centered[i] = values[i] - mean;
        }
        return centered;
    }

}
