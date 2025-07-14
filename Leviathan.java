import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class Leviathan {
    static int dimension = 30;
    static double p = 0.5; 
    static ArrayList<Integer> movesMade = new ArrayList<>();

    public static void main(String[] args) {
        int[] seeds = new int[2];
        for (int i = 0; i < seeds.length; i++) {
            seeds[i] = i + 1;
        }

        
        for (int seed : seeds) {
            runTest(seed);
        }

        System.out.println("Average Moves: " + Stats.computeMean(movesMade));


        //generateTrainingData(3);
    }

    public static void runTest(int seed) {
        Ship ship = Ship.shipGenerator(dimension, p, seed);
        Bot bot = new Bot(ship, seed);
        int maxSteps = 5000;
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

        if (step > maxSteps) {
            System.out.println("Some shit went down, that ain't normal");
            return;
        }

        int open = 0;
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (ship.getCell()[i][j].isOpen()) {
                    open++;
                }
            }
        }

        movesMade.add(bot.getNumberOfMoves());
        System.out.println("Test " + seed + " Finished.");
        System.out.println("Bot took " + bot.getNumberOfMoves() + " moves.");
        System.out.println("Open cells: " + open);
    }

    public static void generateTrainingData(int tests) {
        for (int i = 1; i <= tests; i++) {
            Ship ship = Ship.shipGenerator(dimension, p, i);
            Bot bot = new Bot(ship, i);
            
            try {
                PrintWriter writer = new PrintWriter(new FileWriter("ship_" + i + "_data.csv"));
                writer.println("bx,by,rx,ry,t_value,dimension,open_cells");
                
                int validConfigurations = 0;
                int totalOpenCells = 0;
                
                // Count open cells first
                for (int x = 0; x < dimension; x++) {
                    for (int y = 0; y < dimension; y++) {
                        if (bot.map[x][y].isOpen()) {
                            totalOpenCells++;
                        }
                    }
                }
                
                // Generate all valid configurations
                for (int bx = 0; bx < dimension; bx++) {
                    for (int by = 0; by < dimension; by++) {
                        for (int rx = 0; rx < dimension; rx++) {
                            for (int ry = 0; ry < dimension; ry++) {
                                // Only include valid (non-blocked) cells
                                if (bot.map[bx][by].isOpen() && bot.map[rx][ry].isOpen()) {
                                    double tValue = bot.T[bx][by][rx][ry];
                                    writer.println(bx + "," + by + "," + rx + "," + ry + "," +
                                                tValue + "," + dimension + "," + totalOpenCells);
                                    validConfigurations++;
                                }
                            }
                        }
                    }
                }
                
                writer.close();
                
                System.out.println("Ship " + i + ":");
                System.out.println("  Total open cells: " + totalOpenCells);
                System.out.println("  Valid configurations: " + validConfigurations);
                System.out.println("  Expected (N²): " + (totalOpenCells * totalOpenCells));
                System.out.println("  Actual rows: " + validConfigurations);
                
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // Helper function to print Average, Standard Error, and Required Sample Size
    public static void printStats(ArrayList<Double> pair) {
        //System.out.printf("%-12s %10s %10s\n", "", "Moves", "Senses");
        System.out.printf("%-12s %10.2f %10.2f\n", "Average:", pair.get(0), pair.get(1));
        //System.out.printf("%-12s %10.2f %10.2f\n", "SE:", pair.get(2), pair.get(3));
        System.out.printf("%-12s %10d %10d\n\n", "Required N:", (int)Math.ceil(pair.get(4)), (int)Math.ceil(pair.get(5)));
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
