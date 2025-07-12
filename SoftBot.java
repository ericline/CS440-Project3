import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SoftBot extends Bot {

    private double[][] pkb; // Probabilistic Knowledge base
    private double alpha;   // Sensory sensitivity
    private Random random;  // Random generator to simulate a beep  

    public SoftBot(Ship ship, int seed, double alpha) {
        super(ship, seed);

        this.pkb = new double[dimension][dimension];
        this.alpha = alpha;
        this.random = new Random(seed);

        // Fill PKB
        initializeProbabilites();

        // Place Space Rat
        ship.initializeSpaceRat();
    }

    public void initializeProbabilites() {
        // (1 / number of possible rat cells) or 0 if blocked
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if(map[i][j].isOpen()) {
                    pkb[i][j] = 1.0 / ship.numOfOpenCells;
                }
                else {
                    pkb[i][j] = 0.0;
                }
            }
        }
    }

    @Override
    public boolean sense() {
        
        // If rat is in current cell, return true
        if (map[x][y].ratStatus == RatStatus.HasRat) {
            return true;
        }

        // No rat on cell, so probabliity of cell = 0
        pkb[x][y] = 0.0;

        // Calculate sensor beep probability: e ^ -alpha(k - 1)
        int k = manhattan_distance(map[x][y], ship.getSpaceRat());
        double beepProbabliity = Math.exp(-alpha * (k - 1));

        // Simulate a beep with the probability beepProbability.
        boolean beep = random.nextDouble() < beepProbabliity;

        // Update probabilities
        updateProbabilities(beep);

        return false;
    }

    // Update prior probabilities -> P(rat (i, j) | beep (x, y))
    public void updateProbabilities(boolean beep) {
        double[][] newProbabilities = new double[dimension][dimension];
        double totalProbabilities = 0.0;

        // For each open cell (minus the one bot is at)
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (map[i][j].isOpen() && !(i == x && j == y)) {
                    
                    // Calculating numerator: P(beep (x, y) | rat (i, j))

                    // Distance k from bot to cell
                    int k = manhattan_distance(map[x][y], map[i][j]);

                    // Calculate probability of beep given rat at cell (i, j)
                    // P(beep (x, y) | rat (i, j))
                    double beepProbability = Math.exp(-alpha * (k - 1));
                    double beepGivenRat = beep ? beepProbability : (1 - beepProbability);
                    
                    // P(beep (x, y) | rat (i, j)) * P(rat (i, j))
                    newProbabilities[i][j] = beepGivenRat * pkb[i][j];

                    // Summing up for P(beep)
                    totalProbabilities += newProbabilities[i][j];
                }
                else {
                    newProbabilities[i][j] = 0.0;
                }
            }
        }

        // Update PKB
        if (totalProbabilities > 0) {
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    // If not current
                    if (!(i == x && j == y)) {
                        // P(beep (x, y) | rat (i, j)) * P(rat (i, j)) / P(beep (x, y))
                        pkb[i][j] = newProbabilities[i][j] / totalProbabilities;
                    } else {
                        pkb[i][j] = 0.0;
                    }
                }
            }
        }
    }

    public void executeBasicStrategy() {
        while (true) {
            // Sense at current location
            if (sense()) {
                // System.out.println("Rat found!");
                break;
            }
            
            // Find cell with highest probability
            Cell target = getHighestProbabilityCell();
            if (target == null) break;
            
            // Move to target
            List<Cell> path = findShortestPath(target);
            followPath(path);
        }
    }

    public void executeNewStrategy() {
        int i = 0;
        while (true) {
            // Sense at current location
            if (sense()) {
                // System.out.println("Rat found!");
                break;
            }
            
            // Find cell with highest probability
            Cell target = getHighestUtilityCell();
            if (target == null) break;
            
            // Move to target
            List<Cell> path = findShortestPath(target);
            if (path.size() == 1) {
                System.out.println("Error");
            }
            followPath(path);
            
            i++;
        }
    }

    private Cell getHighestProbabilityCell() {
        double maxProb = -1;
        List<Cell> maxCells = new ArrayList<>();
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (pkb[i][j] > maxProb) {
                    maxProb = pkb[i][j];
                    maxCells.clear();
                    maxCells.add(map[i][j]);
                } else if (pkb[i][j] == maxProb && maxProb > 0) {
                    maxCells.add(map[i][j]);
                }
            }
        }
        
        return maxCells.isEmpty() ? null : maxCells.get(random.nextInt(maxCells.size()));
    }

    private Cell getHighestUtilityCell() {
        double maxUtility = -1;
        List<Cell> bestCells = new ArrayList<>();   
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (map[i][j].isOpen() ) {
                    if (i == x && j == y) continue;
                    int distance = findShortestPath(map[i][j]).size();
                    double neighborSum = getNeighborSum(map[i][j]);
                    double utility = (pkb[i][j] + 0.5 * neighborSum) / (1 + Math.log(distance + 1));
                    if (utility > maxUtility) {
                        maxUtility = utility;
                        bestCells.clear();
                        bestCells.add(map[i][j]);
                    } else if (utility == maxUtility ) {
                        bestCells.add(map[i][j]);
                    }
                }
            }
        }

        return bestCells.isEmpty() ? null : bestCells.get(random.nextInt(bestCells.size()));
    }

    private double getNeighborSum(Cell cell) {
        double sum = 0;
        for (Cell neighbor : cell.neighbors) {
            if(neighbor.isOpen())
                sum += pkb[neighbor.x][neighbor.y];
        }
        return sum;
    }

    
}
