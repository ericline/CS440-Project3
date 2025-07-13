import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Leviathan {
    static int dimension = 30;
    static double p = 0.5; 
    public static void main(String[] args) {
        int[] seeds = new int[20];
        for (int i = 0; i < seeds.length; i++) {
            seeds[i] = i + 1;
        }
        Ship ship = Ship.shipGenerator(dimension, p, seeds[1]);
        Bot bot = new Bot(ship, seeds[1]);
        
        int maxSteps = 1000;
        int step = 0;

        while (step < maxSteps) {
            int ratX = ship.getRatCell().x;
            int ratY = ship.getRatCell().y;

            System.out.printf("Step %d: Bot at (%d,%d), Rat at (%d,%d)\n",
                step, bot.getX(), bot.getY(), ratX, ratY);

            if (bot.getX() == ratX && bot.getY() == ratY) {
                System.out.println("Bot caught rat.");
                break;
            }

            boolean success = bot.makeOptimalMove(ratX, ratY);
            if (!success) {
                break;
            }

            step++;
        }

        System.out.println("Test Finished.");
        System.out.println("Bot took " + bot.getNumberOfMoves() + " moves.");
    }

    private static double averageSum(List<Integer> a, List<Integer> b) {
        if (a.size() != b.size() || a.isEmpty()) return -1;
        double total = 0;
        for (int i = 0; i < a.size(); i++) {
            total += a.get(i) + b.get(i);
        }
        return total / a.size();
    }

    // Helper function to print Average, Standard Error, and Required Sample Size
    public static void printStats(ArrayList<Double> pair) {
        System.out.printf("%-12s %10s %10s\n", "", "Moves", "Senses");
        System.out.printf("%-12s %10.2f %10.2f\n", "Average:", pair.get(0), pair.get(1));
        //System.out.printf("%-12s %10.2f %10.2f\n", "SE:", pair.get(2), pair.get(3));
        //System.out.printf("%-12s %10d %10d\n\n", "Required N:", (int)Math.ceil(pair.get(4)), (int)Math.ceil(pair.get(5)));
        System.out.println();
    }

    // Helper function to add stats to return array
    public static ArrayList<Double> stats(List<Integer> moves, List<Integer> senses) {
        ArrayList<Double> ret = new ArrayList<>();
        ret.add(Stats.computeMean(moves));
        ret.add(Stats.computeMean(senses));
        ret.add(Stats.computeStandardError(moves));
        ret.add(Stats.computeStandardError(senses));
        ret.add(Stats.computeRequiredSampleSize(moves));
        ret.add(Stats.computeRequiredSampleSize(senses));
        return ret;
    }

}
