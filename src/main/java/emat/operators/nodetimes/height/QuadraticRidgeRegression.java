package emat.operators.nodetimes.height;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;

import java.util.Arrays;

/**
 * Ridge regression of a single value on the standardised inputs, their squares and their pairwise products. The features
 * are standardised once more after the expansion, so that the penalty acts on all of them alike. The penalty is chosen
 * from a grid by blocked cross-validation. The fit only needs the Gram matrices of the folds, which are accumulated in
 * a streaming way, so that the expanded features of all samples are never held in memory at once.
 */
public class QuadraticRidgeRegression {

    private static final double MIN_SCALE = 1.0E-12;

    private final int numInputs;
    private final int numFeatures;
    private final double[] penalties;

    private double[] inputMean;
    private double[] inputScale;
    private double[] featureMean;
    private double[] featureScale;
    private double targetMean;
    private double[] coefficients;
    private double penalty;

    /** Creates a regression of the given number of inputs that chooses its penalty from the given grid. */
    public QuadraticRidgeRegression(int numInputs, double[] penalties) {
        this.numInputs = numInputs;
        this.numFeatures = numInputs + numInputs * (numInputs + 1) / 2;
        this.penalties = penalties.clone();
    }

    /** Returns the grid of penalties 10^-3, 10^-2, ..., 10^3. */
    public static double[] createDefaultPenalties() {
        double[] penalties = new double[7];
        for (int i = 0; i < penalties.length; i++) {
            penalties[i] = Math.pow(10.0, i - 3);
        }
        return penalties;
    }

    /**
     * Fits the regression. The folds give the cross-validation fold of every sample, with values from 0 to numFolds - 1.
     * With a single penalty, no cross-validation is needed and the folds are ignored.
     */
    public void fit(double[][] inputs, double[] targets, int[] folds, int numFolds) {
        this.computeStandardisation(inputs, targets);

        double[][][] gramPerFold = new double[numFolds][this.numFeatures][this.numFeatures];
        double[][] momentPerFold = new double[numFolds][this.numFeatures];
        double[] squaredTargetPerFold = new double[numFolds];
        this.accumulateFolds(inputs, targets, folds, gramPerFold, momentPerFold, squaredTargetPerFold);

        double[][] gram = new double[this.numFeatures][this.numFeatures];
        double[] moment = new double[this.numFeatures];
        QuadraticRidgeRegression.sumFolds(gramPerFold, momentPerFold, gram, moment);

        this.penalty = this.penalties.length == 1 ? this.penalties[0]
                : this.choosePenalty(gram, moment, gramPerFold, momentPerFold, squaredTargetPerFold);
        this.coefficients = QuadraticRidgeRegression.solveRidge(gram, moment, this.penalty);
    }

    /** Returns the prediction for the given inputs. */
    public double predict(double[] input) {
        double[] features = this.computeFeatures(input);
        double prediction = this.targetMean;
        for (int j = 0; j < this.numFeatures; j++) {
            prediction += features[j] * this.coefficients[j];
        }
        return prediction;
    }

    /** Returns the penalty that the last fit chose. */
    public double getPenalty() {
        return this.penalty;
    }

    private void computeStandardisation(double[][] inputs, double[] targets) {
        int numSamples = inputs.length;
        this.inputMean = new double[this.numInputs];
        this.inputScale = new double[this.numInputs];
        QuadraticRidgeRegression.computeMoments(inputs, this.inputMean, this.inputScale);

        // the features are standardised with the moments of the expanded standardised inputs, which are computed in two
        // passes without holding the expanded rows
        this.featureMean = new double[this.numFeatures];
        this.featureScale = new double[this.numFeatures];
        for (double[] input : inputs) {
            double[] features = this.expand(this.standardiseInput(input));
            for (int j = 0; j < this.numFeatures; j++) {
                this.featureMean[j] += features[j] / numSamples;
            }
        }
        for (double[] input : inputs) {
            double[] features = this.expand(this.standardiseInput(input));
            for (int j = 0; j < this.numFeatures; j++) {
                this.featureScale[j] += (features[j] - this.featureMean[j]) * (features[j] - this.featureMean[j]) / numSamples;
            }
        }
        for (int j = 0; j < this.numFeatures; j++) {
            this.featureScale[j] = Math.max(Math.sqrt(this.featureScale[j]), MIN_SCALE);
        }

        this.targetMean = Arrays.stream(targets).average().orElse(0.0);
    }

    /**
     * Adds the Gram matrix, the moment with the centred targets and the sum of squared targets of every sample to those
     * of its fold. Only the lower triangle of the Gram matrices is filled.
     */
    private void accumulateFolds(double[][] inputs, double[] targets, int[] folds, double[][][] gramPerFold,
                                 double[][] momentPerFold, double[] squaredTargetPerFold) {
        for (int i = 0; i < inputs.length; i++) {
            double[] features = this.computeFeatures(inputs[i]);
            double target = targets[i] - this.targetMean;
            double[][] gram = gramPerFold[folds[i]];
            double[] moment = momentPerFold[folds[i]];
            for (int a = 0; a < this.numFeatures; a++) {
                double[] row = gram[a];
                double featureA = features[a];
                for (int b = 0; b <= a; b++) {
                    row[b] += featureA * features[b];
                }
                moment[a] += featureA * target;
            }
            squaredTargetPerFold[folds[i]] += target * target;
        }
    }

    private static void sumFolds(double[][][] gramPerFold, double[][] momentPerFold, double[][] gram, double[] moment) {
        for (int k = 0; k < gramPerFold.length; k++) {
            for (int a = 0; a < moment.length; a++) {
                for (int b = 0; b <= a; b++) {
                    gram[a][b] += gramPerFold[k][a][b];
                }
                moment[a] += momentPerFold[k][a];
            }
        }
    }

    /** Returns the penalty with the smallest squared error on the held-out folds. */
    private double choosePenalty(double[][] gram, double[] moment, double[][][] gramPerFold, double[][] momentPerFold,
                                 double[] squaredTargetPerFold) {
        double bestError = Double.POSITIVE_INFINITY;
        double bestPenalty = this.penalties[0];
        for (double candidate : this.penalties) {
            double error = 0.0;
            for (int k = 0; k < gramPerFold.length; k++) {
                double[][] trainGram = QuadraticRidgeRegression.subtract(gram, gramPerFold[k]);
                double[] trainMoment = QuadraticRidgeRegression.subtract(moment, momentPerFold[k]);
                double[] fitted = QuadraticRidgeRegression.solveRidge(trainGram, trainMoment, candidate);
                error += QuadraticRidgeRegression.computeQuadraticForm(fitted, gramPerFold[k])
                        - 2.0 * QuadraticRidgeRegression.computeDot(fitted, momentPerFold[k])
                        + squaredTargetPerFold[k];
            }
            if (error < bestError) {
                bestError = error;
                bestPenalty = candidate;
            }
        }
        return bestPenalty;
    }

    /** Solves (gram + penalty I) w = moment, where only the lower triangle of the gram matrix is filled. */
    private static double[] solveRidge(double[][] gram, double[] moment, double penalty) {
        int n = moment.length;
        double[][] matrix = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b <= a; b++) {
                matrix[a][b] = gram[a][b];
                matrix[b][a] = gram[a][b];
            }
            matrix[a][a] += penalty;
        }
        RealVector solution = new CholeskyDecomposition(new Array2DRowRealMatrix(matrix, false))
                .getSolver().solve(new ArrayRealVector(moment, false));
        return solution.toArray();
    }

    private static double computeQuadraticForm(double[] w, double[][] lowerGram) {
        double sum = 0.0;
        for (int a = 0; a < w.length; a++) {
            for (int b = 0; b < a; b++) {
                sum += 2.0 * w[a] * w[b] * lowerGram[a][b];
            }
            sum += w[a] * w[a] * lowerGram[a][a];
        }
        return sum;
    }

    private static double computeDot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double[][] subtract(double[][] a, double[][] b) {
        double[][] difference = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            difference[i] = QuadraticRidgeRegression.subtract(a[i], b[i]);
        }
        return difference;
    }

    private static double[] subtract(double[] a, double[] b) {
        double[] difference = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            difference[i] = a[i] - b[i];
        }
        return difference;
    }

    private double[] computeFeatures(double[] input) {
        double[] features = this.expand(this.standardiseInput(input));
        for (int j = 0; j < this.numFeatures; j++) {
            features[j] = (features[j] - this.featureMean[j]) / this.featureScale[j];
        }
        return features;
    }

    private double[] standardiseInput(double[] input) {
        double[] standardised = new double[this.numInputs];
        for (int i = 0; i < this.numInputs; i++) {
            standardised[i] = (input[i] - this.inputMean[i]) / this.inputScale[i];
        }
        return standardised;
    }

    /** Returns the values, their squares and their pairwise products. */
    private double[] expand(double[] values) {
        int n = values.length;
        double[] features = new double[this.numFeatures];
        System.arraycopy(values, 0, features, 0, n);

        int index = n;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                features[index++] = values[i] * values[j];
            }
        }
        return features;
    }

    private static void computeMoments(double[][] rows, double[] mean, double[] scale) {
        int numSamples = rows.length;
        for (double[] row : rows) {
            for (int j = 0; j < mean.length; j++) {
                mean[j] += row[j] / numSamples;
            }
        }
        for (double[] row : rows) {
            for (int j = 0; j < mean.length; j++) {
                scale[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / numSamples;
            }
        }
        for (int j = 0; j < mean.length; j++) {
            scale[j] = Math.max(Math.sqrt(scale[j]), MIN_SCALE);
        }
    }

}
