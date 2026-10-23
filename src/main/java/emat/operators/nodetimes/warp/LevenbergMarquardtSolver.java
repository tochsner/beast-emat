package emat.operators.nodetimes.warp;

import org.ojalgo.matrix.decomposition.Cholesky;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.R064Store;

import java.util.function.Function;

/**
 * Minimises the sum of the squared differences between a target and a model by Levenberg-Marquardt steps. Every step
 * solves the damped normal equations with a Cholesky decomposition, which is cheaper than the QR decomposition of the
 * Jacobian that the textbook implementations use when the Jacobian has many more rows than columns.
 */
final class LevenbergMarquardtSolver {

    /** The values of the model at a point, and the derivatives of the values, by row, with respect to the point. */
    record Evaluation(double[] values, double[][] jacobian) {
    }

    /** The point the solver ended at. */
    record Solution(double[] point, int numEvaluations) {
    }

    // the initial damping, relative to the squared norms of the columns of the Jacobian
    private static final double INITIAL_DAMPING = 1e-3;

    // the damping is never lowered to where the damped normal equations cannot be told from the undamped ones, nor raised past where a step is hopeless
    private static final double MIN_DAMPING = 1e-12;
    private static final double MAX_DAMPING = 1e16;

    private final double tolerance;
    private final int maxEvaluations;
    private final Function<double[], Evaluation> model;
    private final double[] target;

    private int numEvaluations;

    // the squared norms of the columns of the Jacobian that the damping is relative to, which only grow, so that the damping of a
    // direction does not vanish where the Jacobian happens to be flat
    private double[] scaling;

    /**
     * Creates a solver for the model, which is fitted to the target. The solver stops when the cost, the point or the
     * cost stops changing by more than the tolerance, as the step and as the model predicts it, or the residual is orthogonal to
     * the columns of the Jacobian up to the tolerance.
     */
    LevenbergMarquardtSolver(Function<double[], Evaluation> model, double[] target, double tolerance, int maxEvaluations) {
        this.model = model;
        this.target = target;
        this.tolerance = tolerance;
        this.maxEvaluations = maxEvaluations;
    }

    /**
     * Returns the point that the solver ends at when it starts from the given point. Throws a runtime exception if the
     * model is not finite at the start, if the solver reaches the maximum number of evaluations, or if no step can be found.
     */
    Solution solve(double[] start) {
        this.numEvaluations = 0;

        double[] point = start.clone();
        Evaluation evaluation = this.evaluate(point);
        double[] residual = this.computeResidual(evaluation);
        double cost = this.computeCost(residual);
        if (!Double.isFinite(cost)) {
            throw new IllegalStateException("the model is not finite at the start");
        }

        int numPoint = point.length;
        this.scaling = new double[numPoint];
        double damping = Double.NaN;
        double dampingFactor = 2.0;
        Cholesky<Double> cholesky = Cholesky.R064.make(numPoint, numPoint);

        while (true) {
            R064Store jacobian = this.createStore(evaluation.jacobian(), numPoint);
            R064Store normalMatrix = R064Store.FACTORY.make(numPoint, numPoint);
            normalMatrix.fillByMultiplying(jacobian.transpose(), jacobian);
            R064Store gradient = R064Store.FACTORY.make(numPoint, 1);
            gradient.fillByMultiplying(jacobian.transpose(), R064Store.wrap(residual));

            if (this.hasReachedStationaryPoint(normalMatrix, gradient, residual, cost)) {
                return new Solution(point, this.numEvaluations);
            }
            this.updateScaling(normalMatrix);
            if (Double.isNaN(damping)) {
                damping = INITIAL_DAMPING;
            }

            // the damping is raised until a step lowers the cost, and lowered after a step that does
            while (true) {
                if (this.numEvaluations >= this.maxEvaluations) {
                    throw new IllegalStateException("maximal count (" + this.maxEvaluations + ") exceeded: evaluations");
                }
                if (damping > MAX_DAMPING) {
                    throw new IllegalStateException("no step lowers the cost");
                }

                double[] step = this.computeStep(cholesky, normalMatrix, gradient, damping);
                if (step == null) {
                    damping *= dampingFactor;
                    dampingFactor *= 2.0;
                    continue;
                }

                double[] newPoint = new double[numPoint];
                for (int i = 0; i < numPoint; i++) {
                    newPoint[i] = point[i] + step[i];
                }
                Evaluation newEvaluation = this.evaluate(newPoint);
                double[] newResidual = this.computeResidual(newEvaluation);
                double newCost = this.computeCost(newResidual);

                double predictedReduction = this.computePredictedReduction(step, gradient, damping);
                double gain = (cost - newCost) / predictedReduction;
                if (!Double.isFinite(newCost) || !(gain > 0.0)) {
                    damping *= dampingFactor;
                    dampingFactor *= 2.0;
                    continue;
                }

                // a step that is small because of the damping says nothing about the cost, so the step itself does not count
                boolean isConverged = cost - newCost <= this.tolerance * cost && predictedReduction <= this.tolerance * cost;
                point = newPoint;
                evaluation = newEvaluation;
                residual = newResidual;
                cost = newCost;

                damping = Math.max(damping * Math.max(1.0 / 3.0, 1.0 - Math.pow(2.0 * gain - 1.0, 3)), MIN_DAMPING);
                dampingFactor = 2.0;
                if (isConverged) {
                    return new Solution(point, this.numEvaluations);
                }
                break;
            }
        }
    }

    private Evaluation evaluate(double[] point) {
        this.numEvaluations++;
        return this.model.apply(point);
    }

    private double[] computeResidual(Evaluation evaluation) {
        double[] residual = new double[this.target.length];
        for (int i = 0; i < residual.length; i++) {
            residual[i] = this.target[i] - evaluation.values()[i];
        }
        return residual;
    }

    private double computeCost(double[] residual) {
        double cost = 0.0;
        for (double value : residual) {
            cost += value * value;
        }
        return 0.5 * cost;
    }

    /** Copies the rows of the Jacobian into a store, which keeps its entries by column. */
    private R064Store createStore(double[][] rows, int numColumns) {
        int numRows = rows.length;
        double[] entries = new double[numRows * numColumns];
        for (int row = 0; row < numRows; row++) {
            for (int column = 0; column < numColumns; column++) {
                entries[column * numRows + row] = rows[row][column];
            }
        }
        return R064Store.wrap(entries, numRows);
    }

    /**
     * Returns whether the residual is orthogonal to the columns of the Jacobian up to the tolerance, which is where the cost cannot
     * be lowered by a step. The cosine of the angle between the residual and a column is its entry of the gradient, divided by the
     * norms of the column, which is the square root of the diagonal entry, and of the residual.
     */
    private boolean hasReachedStationaryPoint(MatrixStore<Double> normalMatrix, MatrixStore<Double> gradient, double[] residual, double cost) {
        if (cost == 0.0) {
            return true;
        }

        double residualNorm = Math.sqrt(2.0 * cost);
        for (int i = 0; i < gradient.countRows(); i++) {
            double columnNorm = Math.sqrt(normalMatrix.doubleValue(i, i));
            if (columnNorm > 0.0 && Math.abs(gradient.doubleValue(i)) > this.tolerance * columnNorm * residualNorm) {
                return false;
            }
        }
        return true;
    }

    /** Raises the squared norms of the columns to those of the Jacobian, where a column of zeros counts as one. */
    private void updateScaling(MatrixStore<Double> normalMatrix) {
        for (int i = 0; i < this.scaling.length; i++) {
            double squaredNorm = normalMatrix.doubleValue(i, i);
            this.scaling[i] = Math.max(this.scaling[i], squaredNorm > 0.0 ? squaredNorm : 1.0);
        }
    }

    /**
     * Solves the normal equations, whose diagonal entries are raised by the damping times the scaling, for the step. Returns null if the damped
     * matrix is not positive definite.
     */
    private double[] computeStep(Cholesky<Double> cholesky, R064Store normalMatrix, R064Store gradient, double damping) {
        int numPoint = (int) normalMatrix.countRows();
        R064Store damped = normalMatrix.copy();
        for (int i = 0; i < numPoint; i++) {
            damped.add(i, i, damping * this.scaling[i]);
        }

        if (!cholesky.decompose(damped) || !cholesky.isSolvable()) {
            return null;
        }

        MatrixStore<Double> solution = cholesky.getSolution(gradient);
        double[] step = new double[numPoint];
        for (int i = 0; i < numPoint; i++) {
            step[i] = solution.doubleValue(i);
            if (!Double.isFinite(step[i])) {
                return null;
            }
        }
        return step;
    }

    /** Returns the decrease of the cost that the quadratic model predicts for the step, which is positive for a step with a positive gain. */
    private double computePredictedReduction(double[] step, MatrixStore<Double> gradient, double damping) {
        double reduction = 0.0;
        for (int i = 0; i < step.length; i++) {
            double dampedStep = damping * this.scaling[i] * step[i];
            reduction += step[i] * (dampedStep + gradient.doubleValue(i));
        }
        return 0.5 * reduction;
    }

}
