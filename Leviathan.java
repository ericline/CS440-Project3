import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Leviathan {
    static int dimension = 30;
    static double p = 0.5; 
    public static void main(String[] args) {
    int[] seeds = new int[1];
    for (int i = 0; i < seeds.length; i++) {
        seeds[i] = i + 1;
    }

    int sensingRange = 15;
    double alpha = 0.1;

    List<Integer> movesOriginal = new ArrayList<>();
    List<Integer> sensesOriginal = new ArrayList<>();
    List<Integer> movesPlus = new ArrayList<>();
    List<Integer> sensesPlus = new ArrayList<>();
    List<Integer> movesSoftBasic = new ArrayList<>();
    List<Integer> sensesSoftBasic = new ArrayList<>();
    List<Integer> movesSoftNew = new ArrayList<>();
    List<Integer> sensesSoftNew = new ArrayList<>();

    System.out.println("Testing...");
    for (int seed : seeds) {
        // Hard Sensor Bot - Original Strategy
        Ship ship1 = Ship.shipGenerator(dimension, p, seed);
        HardSensorBot bot1 = new HardSensorBot(ship1, sensingRange);
        int[] resultOriginal = SearchByHardSensor(bot1, ship1);
        if (resultOriginal != null) {
            movesOriginal.add(resultOriginal[0]);
            sensesOriginal.add(resultOriginal[1]);
        }

        // Hard Sensor Bot - Utility Strategy
        Ship ship2 = Ship.shipGenerator(dimension, p, seed);
        HardSensorBot bot2 = new HardSensorBot(ship2, sensingRange);
        int[] resultPlus = SearchByHardSensorVersion2(bot2, ship2);
        if (resultPlus != null) {
            movesPlus.add(resultPlus[0]);
            sensesPlus.add(resultPlus[1]);
        }

        // Soft Sensor Bot - Basic Strategy
        Ship softShip1 = Ship.shipGenerator(dimension, p, seed);
        SoftBot softBot1 = new SoftBot(softShip1, seed, alpha);
        int[] softBasic = SearchBySoftSensorBasic(softBot1);
        if (softBasic != null) {
            movesSoftBasic.add(softBasic[0]);
            sensesSoftBasic.add(softBasic[1]);
        }

        // Soft Sensor Bot - Utility Strategy
        Ship softShip2 = Ship.shipGenerator(dimension, p, seed);
        SoftBot softBot2 = new SoftBot(softShip2, seed, alpha);
        int[] softNew = SearchBySoftSensorNew(softBot2);
        if (softNew != null) {
          movesSoftNew.add(softNew[0]);
          sensesSoftNew.add(softNew[1]);
        }
        System.out.println("Test " + seed + " Complete.");
    }

    System.out.println("Hard Sensor Original Strategy:");
    printStats(stats(movesOriginal, sensesOriginal));

    System.out.println("Hard Sensor Utility Strategy:");
        printStats(stats(movesPlus, sensesPlus));

    System.out.println("Soft Sensor Basic Strategy:");
        printStats(stats(movesSoftBasic, sensesSoftBasic));

    System.out.println("Soft Sensor Utility Strategy:");
        printStats(stats(movesSoftNew, sensesSoftNew));

    System.out.println("Running Optimal Range Tests...");
    //runSensingRangeExperiment();
    runSensitivityRangeExperiment();
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

    /*
    * 1. initialize the probability array
    * 2. pick a random open cell as the starting point, run the sensor
    * 3. if the rat is not found, move to the closest cell  whose probability is not 0 
    *     by finding a shortest path from the current location to the target cell, run the sensor
    * 4. if the rat is found, return the cell
    * 5. keep track of the number of moves and sense actions
    */

    public static int[] SearchByHardSensor(HardSensorBot bot, Ship ship) {
        bot.initialProbabilityMap();
        Cell currentCell = bot.pickInitialCell();

        int moves = 0;
        int senseActions = 0;

        while (true) {
            // First check if the rat is at the current cell
            if (currentCell.getX() == ship.getRatCell().getX() &&
                currentCell.getY() == ship.getRatCell().getY()) {
                return new int[] { moves, senseActions };
            }

            // Run the hard sensor at the current cell
            HardSensorBot.Status status = bot.HardSensing(currentCell);
            senseActions++;

            bot.updateProbabilityMap(currentCell, status);

            Cell nextCell = bot.pickClosestPossibleCell(currentCell);
            if (nextCell == null) {
                System.err.println("No cell with non-zero probability found. Ending...");
                return null;
            }

            if (!nextCell.equals(currentCell)) {
                List<Cell> path = bot.findShortestPath(currentCell, nextCell);
                for (int i = 1; i < path.size(); i++) {
                    currentCell = path.get(i);
                    moves++;
                }
            }
        }
    }


    public static int[] SearchByHardSensorVersion2(HardSensorBot bot, Ship ship) {
        bot.initialProbabilityMap();
        Cell currentCell = bot.pickInitialCell();

        int moves = 0;
        int senseActions = 0;

        while (true) {
            // First check if the rat is at the current cell
            if (currentCell.getX() == ship.getRatCell().getX() &&
                currentCell.getY() == ship.getRatCell().getY()) {
                return new int[] { moves, senseActions };
            }

            // Run the hard sensor at the current cell
            HardSensorBot.Status status = bot.HardSensing(currentCell);
            senseActions++;

            bot.updateProbabilityMap(currentCell, status);

            Cell nextCell = bot.pickBestUtilityCellVersion2(currentCell);
            if (nextCell == null) {
                System.err.println("No cell with non-zero probability found. Ending...");
                return null;
            }

            if (!nextCell.equals(currentCell)) {
                List<Cell> path = bot.findShortestPath(currentCell, nextCell);
                for (int i = 1; i < path.size(); i++) {
                    currentCell = path.get(i);
                    moves++;
                }
            }
        }
    }

    public static int[] SearchByHardSensorPlus(HardSensorBot bot, Ship ship) {
        bot.initialProbabilityMap();
        Cell currentCell = bot.pickInitialCell();

        int moves = 0;
        int senseActions = 0;
        Set<Cell> covered = new HashSet<>();
        boolean[][] visited = new boolean[ship.getDimension()][ship.getDimension()];

        while (true) {
            if (moves > 3000 || senseActions > 3000) {
                System.err.println("infinite loop?");
                return null;
            }

            // Check if rat is at current cell
            if (currentCell.getX() == ship.getRatCell().getX() &&
                currentCell.getY() == ship.getRatCell().getY()) {
                return new int[] { moves, senseActions };
            }

            HardSensorBot.Status status = bot.HardSensing(currentCell);
            senseActions++;

            visited[currentCell.getX()][currentCell.getY()] = true;

            int x = currentCell.getX();
            int y = currentCell.getY();
            for (int i = 0; i < ship.getDimension(); i++) {
                for (int j = 0; j < ship.getDimension(); j++) {
                    if (Math.abs(i - x) + Math.abs(j - y) <= bot.getSensoringRange() &&
                        ship.getCell()[i][j].isOpen()) {
                        covered.add(ship.getCell()[i][j]);
                    }
                }
            }

            bot.updateProbabilityMap(currentCell, status);

            boolean isNegativeCase = (status == HardSensorBot.Status.NEGATIVE);
            Cell nextCell = bot.selectNextSensingCenter(currentCell, visited, isNegativeCase);

            if (nextCell == null) {
                nextCell = bot.pickClosestPossibleCell(currentCell);
            }

            if (nextCell == null) {
                System.err.println("no target found? not even closest cell?");
                return null;
            }

            if (!nextCell.equals(currentCell)) {
                List<Cell> path = bot.findShortestPath(currentCell, nextCell);
                for (int i = 1; i < path.size(); i++) {
                    currentCell = path.get(i);
                    moves++;
                }
            }
        }
    }

    public static int[] SearchBySoftSensorBasic(SoftBot bot) {
        bot.executeBasicStrategy();
        return new int[] {bot.movesTaken, bot.senseActions};
    }

    public static int[] SearchBySoftSensorNew(SoftBot bot) {
        bot.executeNewStrategy();
        return new int[] {bot.movesTaken, bot.senseActions};
    }

    public static void runSensingRangeExperiment() {
        int[] sensingRanges = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21};
        int[] seeds = new int[300];
        for (int i = 0; i < seeds.length; i++) seeds[i] = i + 1;

        for (int range : sensingRanges) {
            List<Integer> moves = new ArrayList<>();
            List<Integer> senses = new ArrayList<>();

            for (int seed : seeds) {
                Ship ship = Ship.shipGenerator(dimension, p, seed);
                HardSensorBot bot = new HardSensorBot(ship, range);
                int[] result = SearchByHardSensorVersion2(bot, ship);
                if (result != null) {
                    moves.add(result[0]);
                    senses.add(result[1]);
                }
            }

            double avgMoves = Stats.computeMean(moves);
            double avgSenses = Stats.computeMean(senses);
            double avgTotal = averageSum(moves, senses);
            System.out.printf("%-15d %-10.2f %-10.2f %-15.2f\n", range, avgMoves, avgSenses, avgTotal);
            
        }
    }

    public static void runSensitivityRangeExperiment() {
        double[] sensitivityRanges = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
        int[] seeds = new int[300];
        for (int i = 0; i < seeds.length; i++) seeds[i] = i + 1;

        for (double a : sensitivityRanges) {

            List<Integer> moves = new ArrayList<>();
            List<Integer> senses = new ArrayList<>();
            for (int seed : seeds) {
                Ship ship = Ship.shipGenerator(dimension, p, seed);
                SoftBot bot = new SoftBot(ship, seed, a);
                int[] result = SearchBySoftSensorNew(bot);
                if (result != null) {
                    moves.add(result[0]);
                    senses.add(result[1]);
                }
            }

            double avgMoves = Stats.computeMean(moves);
            double avgSenses = Stats.computeMean(senses);
            double avgTotal = averageSum(moves, senses);
            System.out.printf("%-15.2f %-10.2f %-10.2f %-15.2f\n", a, avgMoves, avgSenses, avgTotal);
        }
    }

}
