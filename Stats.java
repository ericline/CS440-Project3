import java.util.List;

public class Stats {
    public static double computeMean(List<Integer> data) {
        double sum = 0;
        for (int value : data) {
            sum += value;
        }
        return sum / data.size();
    }

    public static double computeVariance(List<Integer> data) {
        double mean = computeMean(data);
        double sumSquaredDiff = 0;

        for (int value : data) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }

        return sumSquaredDiff / (data.size() - 1);
    }

      public static double computeStandardDeviation(List<Integer> data) {
        return Math.sqrt(computeVariance(data));
    }

    public static double computeStandardError(List<Integer> data) {
        return computeStandardDeviation(data) / Math.sqrt(data.size());
    }

    public static double computeRequiredSampleSize(List<Integer> data) {
        // n = ((z * sd) / E) ^2
        double z_score = 1.96; // 95% confidence
        double E = 0.05 * computeMean(data); // acceptable margin of error

        double n = (z_score * computeStandardDeviation(data)) / E;
        return n*n;
    }
}
