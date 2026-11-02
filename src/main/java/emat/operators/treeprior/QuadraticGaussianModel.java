package emat.operators.treeprior;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;

import java.util.Random;

/**
 * A Gaussian distribution of a vector of values whose mean is a quadratic function of a vector of conditions and whose
 * covariance is constant. The mean is fitted by ridge regression on the standardised conditions, their squares and their
 * products, with the penalty chosen by leave-one-out cross-validation, and the covariance is that of the leave-one-out
 * residuals, which does not underestimate the spread the way the residuals of the fit itself do.
 */
public class QuadraticGaussianModel {

    private static final double[] PENALTIES = QuadraticGaussianModel.createPenalties();
    private static final double MIN_SCALE = 1.0E-12;
    private static final double MIN_JITTER = 1.0E-12;
    private static final double RELATIVE_JITTER = 1.0E-9;
    private static final int MAX_CHOLESKY_ATTEMPTS = 12;

    private final int numConditions;
    private final int numValues;
    private final Random random = new Random();

    private double[] conditionMean;
    private double[] conditionScale;
    private double[] featureMean;
    private double[] featureScale;
    private double[][] coefficients;
    private double[] valueMean;
    private double[][] covarianceFactor;
    private double logDeterminantFactor;

    public QuadraticGaussianModel(int numConditions, int numValues) {
        this.numConditions = numConditions;
        this.numValues = numValues;
    }

    private static double[] createPenalties() {
        double[] penalties = new double[20];
        for (int i = 0; i < penalties.length; i++) {
            penalties[i] = Math.pow(10.0, -1.0 + 5.0 * i / (penalties.length - 1));
        }
        return penalties;
    }

    /** Fits the model to the given pairs of conditions and values. */
    public void fit(double[][] conditions, double[][] values) {
        int numSamples = conditions.length;

        this.conditionMean = new double[this.numConditions];
        this.conditionScale = new double[this.numConditions];
        QuadraticGaussianModel.computeMoments(conditions, this.conditionMean, this.conditionScale);

        double[][] features = new double[numSamples][];
        for (int i = 0; i < numSamples; i++) {
            features[i] = this.expand(this.standardiseConditions(conditions[i]));
        }
        this.featureMean = new double[features[0].length];
        this.featureScale = new double[features[0].length];
        QuadraticGaussianModel.computeMoments(features, this.featureMean, this.featureScale);
        for (double[] row : features) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (row[j] - this.featureMean[j]) / this.featureScale[j];
            }
        }

        this.valueMean = new double[this.numValues];
        double[][] centered = new double[numSamples][this.numValues];
        for (double[] row : values) {
            for (int k = 0; k < this.numValues; k++) {
                this.valueMean[k] += row[k] / numSamples;
            }
        }
        for (int i = 0; i < numSamples; i++) {
            for (int k = 0; k < this.numValues; k++) {
                centered[i][k] = values[i][k] - this.valueMean[k];
            }
        }

        this.fitRidge(features, centered);
    }

    /**
     * Fits the coefficients of the ridge regression for every penalty from one singular value decomposition, keeps the
     * penalty with the smallest leave-one-out error, and takes the covariance of the leave-one-out residuals.
     */
    private void fitRidge(double[][] features, double[][] centered) {
        int numSamples = features.length;
        int numFeatures = features[0].length;

        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(features, false));
        RealMatrix u = svd.getU();
        RealMatrix v = svd.getV();
        double[] singularValues = svd.getSingularValues();
        int rank = singularValues.length;
        double[][] projected = u.transpose().multiply(new Array2DRowRealMatrix(centered, false)).getData();

        double bestError = Double.POSITIVE_INFINITY;
        double[][] bestResiduals = null;
        double bestPenalty = PENALTIES[0];
        for (double penalty : PENALTIES) {
            double[][] residuals = QuadraticGaussianModel.computeLeaveOneOutResiduals(u, singularValues, projected, centered, penalty);
            double error = 0.0;
            for (double[] row : residuals) {
                for (double value : row) {
                    error += value * value;
                }
            }
            if (error < bestError) {
                bestError = error;
                bestResiduals = residuals;
                bestPenalty = penalty;
            }
        }

        this.coefficients = new double[numFeatures][this.numValues];
        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < this.numValues; k++) {
                double sum = 0.0;
                for (int r = 0; r < rank; r++) {
                    sum += v.getEntry(j, r) * singularValues[r] / (singularValues[r] * singularValues[r] + bestPenalty) * projected[r][k];
                }
                this.coefficients[j][k] = sum;
            }
        }

        this.setCovariance(bestResiduals, numSamples);
    }

    /** Returns the residuals of every sample when it is left out of the fit with the given penalty. */
    private static double[][] computeLeaveOneOutResiduals(RealMatrix u, double[] singularValues, double[][] projected, double[][] centered, double penalty) {
        int numSamples = u.getRowDimension();
        int numValues = centered[0].length;
        int rank = singularValues.length;

        double[] shrinkage = new double[rank];
        for (int r = 0; r < rank; r++) {
            shrinkage[r] = singularValues[r] * singularValues[r] / (singularValues[r] * singularValues[r] + penalty);
        }

        double[][] residuals = new double[numSamples][numValues];
        for (int i = 0; i < numSamples; i++) {
            double leverage = 0.0;
            for (int r = 0; r < rank; r++) {
                leverage += u.getEntry(i, r) * u.getEntry(i, r) * shrinkage[r];
            }
            for (int k = 0; k < numValues; k++) {
                double fitted = 0.0;
                for (int r = 0; r < rank; r++) {
                    fitted += u.getEntry(i, r) * shrinkage[r] * projected[r][k];
                }
                residuals[i][k] = (centered[i][k] - fitted) / Math.max(1.0 - leverage, MIN_SCALE);
            }
        }
        return residuals;
    }

    private void setCovariance(double[][] residuals, int numSamples) {
        double[][] covariance = new double[this.numValues][this.numValues];
        for (double[] row : residuals) {
            for (int a = 0; a < this.numValues; a++) {
                for (int b = 0; b < this.numValues; b++) {
                    covariance[a][b] += row[a] * row[b] / numSamples;
                }
            }
        }
        this.covarianceFactor = QuadraticGaussianModel.computeCholeskyFactor(covariance);

        this.logDeterminantFactor = 0.0;
        for (int k = 0; k < this.numValues; k++) {
            this.logDeterminantFactor += Math.log(this.covarianceFactor[k][k]);
        }
    }

    /** Returns the lower Cholesky factor, adding jitter to the diagonal until the covariance is positive definite. */
    private static double[][] computeCholeskyFactor(double[][] covariance) {
        int n = covariance.length;
        double trace = 0.0;
        for (int i = 0; i < n; i++) {
            trace += covariance[i][i];
        }

        double jitter = Math.max(MIN_JITTER, RELATIVE_JITTER * trace / n);
        for (int attempt = 0; attempt < MAX_CHOLESKY_ATTEMPTS; attempt++) {
            double[][] factor = new double[n][n];
            boolean isPositive = true;
            for (int i = 0; i < n && isPositive; i++) {
                for (int j = 0; j <= i; j++) {
                    double sum = covariance[i][j] + (i == j ? jitter : 0.0);
                    for (int k = 0; k < j; k++) {
                        sum -= factor[i][k] * factor[j][k];
                    }
                    if (i == j) {
                        isPositive = sum > 0.0;
                        factor[i][i] = isPositive ? Math.sqrt(sum) : 0.0;
                        if (!isPositive) {
                            break;
                        }
                    } else {
                        factor[i][j] = sum / factor[j][j];
                    }
                }
            }
            if (isPositive) {
                return factor;
            }
            jitter *= 10.0;
        }
        throw new IllegalStateException("The covariance is not positive definite.");
    }

    /** Returns the mean of the values given the conditions. */
    public double[] computeMean(double[] conditions) {
        double[] features = this.expand(this.standardiseConditions(conditions));
        double[] mean = this.valueMean.clone();
        for (int j = 0; j < features.length; j++) {
            double feature = (features[j] - this.featureMean[j]) / this.featureScale[j];
            for (int k = 0; k < this.numValues; k++) {
                mean[k] += feature * this.coefficients[j][k];
            }
        }
        return mean;
    }

    /** Draws values given the conditions, with the covariance multiplied by the given factor. */
    public double[] sample(double[] conditions, double covarianceScale) {
        double[] values = this.computeMean(conditions);
        double[] noise = new double[this.numValues];
        for (int k = 0; k < this.numValues; k++) {
            noise[k] = this.random.nextGaussian();
        }

        double standardDeviationScale = Math.sqrt(covarianceScale);
        for (int a = 0; a < this.numValues; a++) {
            for (int b = 0; b <= a; b++) {
                values[a] += standardDeviationScale * this.covarianceFactor[a][b] * noise[b];
            }
        }
        return values;
    }

    /** Returns the log density of the values given the conditions, with the covariance multiplied by the given factor. */
    public double computeLogDensity(double[] conditions, double[] values, double covarianceScale) {
        double[] mean = this.computeMean(conditions);

        // solve the lower triangular system for the whitened difference
        double[] whitened = new double[this.numValues];
        double squaredNorm = 0.0;
        for (int a = 0; a < this.numValues; a++) {
            double sum = values[a] - mean[a];
            for (int b = 0; b < a; b++) {
                sum -= this.covarianceFactor[a][b] * whitened[b];
            }
            whitened[a] = sum / this.covarianceFactor[a][a];
            squaredNorm += whitened[a] * whitened[a];
        }

        double logDeterminant = 2.0 * this.logDeterminantFactor + this.numValues * Math.log(covarianceScale);
        return -0.5 * (this.numValues * Math.log(2.0 * Math.PI) + logDeterminant + squaredNorm / covarianceScale);
    }

    private double[] standardiseConditions(double[] conditions) {
        double[] standardised = new double[this.numConditions];
        for (int i = 0; i < this.numConditions; i++) {
            standardised[i] = (conditions[i] - this.conditionMean[i]) / this.conditionScale[i];
        }
        return standardised;
    }

    /** Returns the values, their squares and their pairwise products. */
    private double[] expand(double[] values) {
        int n = values.length;
        double[] features = new double[n + n * (n + 1) / 2];
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
