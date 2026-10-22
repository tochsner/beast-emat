package emat.helper;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Simulates trees of the exponential growth coalescent with all tips at height zero, and gives their exact log density. */
public class CoalescentSimulation {

    public final double[] tipHeights;
    public final double[] internalHeights;
    public final String newick;
    public final double logDensity;

    public CoalescentSimulation(Random random, int numTips, double popSize, double growthRate) {
        this.tipHeights = new double[numTips];
        this.internalHeights = new double[numTips - 1];

        List<String> names = new ArrayList<>();
        List<Double> heights = new ArrayList<>();
        for (int i = 0; i < numTips; i++) {
            names.add("t" + i);
            heights.add(0.0);
        }

        double logDensity = 0.0;
        double time = 0.0;
        double previousHeight = 0.0;
        String lastNewick = "";
        for (int k = numTips; k >= 2; k--) {
            double pairs = k * (k - 1) / 2.0;
            time += -Math.log(random.nextDouble()) / pairs;
            double height = Math.log1p(growthRate * popSize * time) / growthRate;
            this.internalHeights[numTips - k] = height;

            logDensity += Math.log(pairs) + growthRate * height - Math.log(popSize)
                    - pairs * (Math.expm1(growthRate * height) - Math.expm1(growthRate * previousHeight)) / (growthRate * popSize);
            previousHeight = height;

            int first = random.nextInt(k);
            int second = random.nextInt(k - 1);
            if (second >= first) {
                second++;
            }
            String merged = "(" + names.get(first) + ":" + (height - heights.get(first)) + "," + names.get(second) + ":" + (height - heights.get(second)) + ")";
            int low = Math.min(first, second);
            int high = Math.max(first, second);
            names.remove(high);
            heights.remove(high);
            names.set(low, merged);
            heights.set(low, height);
            lastNewick = merged;
        }

        this.newick = lastNewick + ";";
        this.logDensity = logDensity;
    }

}
