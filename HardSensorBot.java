import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

public class HardSensorBot {

    Cell[][] cells;
    int dimension;
    Cell ratCell;
    int sensoringRange;
    double[][] probabilityMap;

    enum Status {
    POSITIVE,
    NEGATIVE
}


    // Constructor
    public HardSensorBot(Ship ship, int k) {
        this.cells = ship.getCells();
        this.dimension = ship.getDimension();
        this.ratCell = ship.getRatCell();
        this.sensoringRange = k;
    }

    public void initialProbabilityMap() {
    this.probabilityMap = new double[dimension][dimension];

    // Count open cells
    int openCellCount = 0;
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (cells[i][j].isOpen()) {
                openCellCount++;
            }
        }
    }

    // Assign uniform probability to open cells
    double uniformProb = 1.0 / openCellCount;
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            probabilityMap[i][j] = cells[i][j].isOpen() ? uniformProb : 0.0;
        }
    }
}


    public Cell pickInitialCell() {
        // Randomly pick an open cell as the initial cell
        int r, c;
        do {
            r = (int) (Math.random() * dimension);
            c = (int) (Math.random() * dimension);
        } while (!cells[r][c].isOpen());
        
        return cells[r][c];
    }

    public Cell pickClosestPossibleCell(Cell currentCell) {
    // Find the closest cell with non-zero probability using Manhattan distance
    double minDistance = Double.MAX_VALUE;
    Cell closestCell = null;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (probabilityMap[i][j] > 0 && cells[i][j].isOpen()) {
                 int distance = Math.abs(currentCell.getX() - i) + Math.abs(currentCell.getY() - j);
                // double distance = Math.sqrt(Math.pow(currentCell.getX() - i, 2) + Math.pow(currentCell.getY() - j, 2));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCell = cells[i][j];
                }
            }
        }
    }

    return closestCell;
}


    /*
     * Pretty bad :(
     */
    public Cell pickHighestProbabilityCell() {
    Cell best = null;
    double maxProb = -1.0;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (!cells[i][j].isOpen()) continue;
            if (probabilityMap[i][j] > maxProb) {
                maxProb = probabilityMap[i][j];
                best = cells[i][j];
            }
        }
    }

    if (best == null) {
        System.err.println("No valid cell found in pickHighestProbabilityCell.");
    }

    return best;
}

    public Cell pickBestUtilityCellVersion2(Cell currentCell) {
    Cell bestCell = null;
    double bestUtility = - 10000000;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (!cells[i][j].isOpen() || probabilityMap[i][j] == 0.0)
                continue;

            Cell candidate = cells[i][j];

            // Use Manhattan distance as path estimate
            int distance = Math.abs(currentCell.getX() - i) + Math.abs(currentCell.getY() - j);
            List<Cell> path = findShortestPath(currentCell, candidate);
            if (path == null || path.size() == 0) continue;

            // Avoid division by zero/log(0)
            double utility =  probabilityMap[i][j] -(0.1* path.size() );

            if (utility > bestUtility) {
                bestUtility = utility;
                bestCell = candidate;
            }
        }
    }

    if (bestCell == null) {
        System.err.println("No such target exists");
    }

    return bestCell;
}



    public Cell pickBestUtilityCellDiv(Cell currentCell) {
    Cell bestCell = null;
    double bestUtility = -1.0;

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (!cells[i][j].isOpen() || probabilityMap[i][j] == 0.0)
                continue;

            Cell candidate = cells[i][j];

            // Use Manhattan distance as path estimate
            int distance = Math.abs(currentCell.getX() - i) + Math.abs(currentCell.getY() - j);

            // Avoid division by zero/log(0)
            double utility = probabilityMap[i][j] / (1.0 + Math.log(1.0 + distance));

            if (utility > bestUtility) {
                bestUtility = utility;
                bestCell = candidate;
            }
        }
    }

    if (bestCell == null) {
        System.err.println("No such target exists");
    }

    return bestCell;
}


    public Status HardSensing(Cell currentCell) {
        // System.out.println("HardSensing at (" + currentCell.getX() + ", " + currentCell.getY() + ")");
        // Simulate hard sensing

            if (Math.abs(currentCell.getX() - ratCell.getX()) + Math.abs(currentCell.getY() - ratCell.getY()) <= sensoringRange) {
            return Status.POSITIVE; // if rat within manhattan distance k
            }
            else {
                return Status.NEGATIVE; // if rat not within manhattan distance k
            }  
    
    }


    public void updateProbabilityMap(Cell currentCell, Status status) {
        // Update the probability map based on the sensing result

        /*
        * If the signal at (a, b) is POSITIVE, 
        * - set the current cell's probability to 0.
        * - recalculate the probabilities of the neighboring cells (i, j) within the sensoring range by conditional probability: 
        * P(rat at (i, j) | positive signal at (a, b) ) 
        *   = P(rat at (i, j), positive signal at (a, b) ) / P(positive signal at (a, b))
        *   = P(rat at (i, j)) * P(positive signal at (a, b) | rat at (i, j)) / P(positive signal at (a, b))
        * 
        * P(positive signal at (a, b) | rat at (i, j))
        *   = 0 if (a, b) == (i, j) or (i, j) is not within the sensoring range
        *   = 1 otherwise
        * 
        * - if the cell (i, j) is not within the sensoring range,
        *   set the probability of the cell (i, j) to 0.
        */

        /*
        * Positive signal at (a, b)
        */
        if (status == Status.POSITIVE) {
            int a = currentCell.getX();
            int b = currentCell.getY();

            double total = 0.0;

            // Zero out invalid cells and compute total mass in valid region
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    if (!cells[i][j].isOpen()) {
                        probabilityMap[i][j] = 0.0;
                        continue;
                    }

                    int manhattan = Math.abs(i - a) + Math.abs(j - b);

                    if ((i == a && j == b) || manhattan > sensoringRange) {
                        probabilityMap[i][j] = 0.0;
                    } else {
                        total += probabilityMap[i][j];
                    }
                }
            }

            // Normalize remaining probabilities
            if (total == 0.0) {
                System.err.println("Errorrrr1");
                return;
            }

            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    if (probabilityMap[i][j] > 0.0) {
                        probabilityMap[i][j] /= total;
                    }
                }
            }

            return;
        }

        /*
        * Negative signal at (a, b)
        */
        if (status == Status.NEGATIVE) {
            int a = currentCell.getX();
            int b = currentCell.getY();

            double[][] newMap = new double[dimension][dimension];
            double total = 0.0;

            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    if (!cells[i][j].isOpen()) {
                        newMap[i][j] = 0.0;
                        continue;
                    }

                    int manhattan = Math.abs(i - a) + Math.abs(j - b);
                    if (manhattan <= sensoringRange) {
                        // Within sensing range => rat cannot be here
                        newMap[i][j] = 0.0;
                    } else {
                        // Outside sensing range => keep original probability
                        newMap[i][j] = probabilityMap[i][j];
                        total += newMap[i][j];
                    }
                }
            }

            if (total == 0) {
                System.err.println("error2");
            } else {
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        probabilityMap[i][j] = newMap[i][j] / total;
                    }
                }
            }

            return;
        }
    }



    public List<Cell> findShortestPath(Cell start, Cell goal) {
        PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingInt(n -> n.totalCost));
        Set<Cell> closedSet = new HashSet<>();
        Map<Cell, Cell> cameFrom = new HashMap<>();
        Map<Cell, Integer> costFromStart = new HashMap<>();

        openSet.add(new Node(start, 0, heuristic(start, goal)));
        costFromStart.put(start, 0);

        while (!openSet.isEmpty()) {
            Node current = openSet.poll();
            Cell currentCell = current.cell;

            if (currentCell.equals(goal)) {
                return reconstructPath(cameFrom, goal);
            }

            closedSet.add(currentCell);

            for (Cell neighbor : getNeighbors(currentCell)) {
                if (!neighbor.isOpen() || closedSet.contains(neighbor)) continue;

                int newCost = costFromStart.get(currentCell) + 1;
                if (!costFromStart.containsKey(neighbor) || newCost < costFromStart.get(neighbor)) {
                    costFromStart.put(neighbor, newCost);
                    cameFrom.put(neighbor, currentCell);
                    int totalCost = newCost + heuristic(neighbor, goal);
                    openSet.add(new Node(neighbor, newCost, totalCost));
                }
            }
        }

        return null; // No path found
    }

    private List<Cell> getNeighbors(Cell cell) {
        List<Cell> neighbors = new ArrayList<>();
        int x = cell.getX();
        int y = cell.getY();

        if (x > 0) neighbors.add(cells[x - 1][y]);
        if (x < dimension - 1) neighbors.add(cells[x + 1][y]);
        if (y > 0) neighbors.add(cells[x][y - 1]);
        if (y < dimension - 1) neighbors.add(cells[x][y + 1]);

        return neighbors;
    }

    private int heuristic(Cell a, Cell b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private List<Cell> reconstructPath(Map<Cell, Cell> cameFrom, Cell goal) {
        List<Cell> path = new ArrayList<>();
        Cell current = goal;
        while (current != null) {
            path.add(current);
            current = cameFrom.get(current);
        }
        Collections.reverse(path);
        return path;
    }

    private static class Node {
        Cell cell;
        int costFromStart;
        int totalCost;

        Node(Cell cell, int costFromStart, int totalCost) {
            this.cell = cell;
            this.costFromStart = costFromStart;
            this.totalCost = totalCost;
        }
    }


    public Cell selectNextSensingCenter(Cell currentCell, boolean[][] visited, boolean isNegativeCase) {

        Set<Cell> covered = new HashSet<>();
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (probabilityMap[i][j] == 0.0 || !cells[i][j].isOpen()) continue;
                int dist = Math.abs(currentCell.getX() - i) + Math.abs(currentCell.getY() - j);
                if (dist <= sensoringRange) {
                    covered.add(cells[i][j]);
                }
            }
        }

        Cell bestCandidate = null;

        // POSITIVE case: rat is within the sensing range
        // pick the furthest (in manhattan dis) open, reachable cell within range, and it is not recently visited
        if (!isNegativeCase) {
            int maxDist = -1;
            int minCost = Integer.MAX_VALUE;

            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    Cell candidate = cells[i][j];
                    if (!candidate.isOpen() || visited[i][j]) continue;

                    // manhattan distance to current cell
                    int dist = Math.abs(i - currentCell.getX()) + Math.abs(j - currentCell.getY());
                    if (dist <= sensoringRange && dist > 0) {
                        List<Cell> path = findShortestPath(currentCell, candidate);
                        if (path == null) continue;
                        int cost = path.size();

                        // find the best candidate where it is the furthest away from current cell
                        // and if there is a tie, pick the one with the lowest cost
                        if (dist > maxDist || (dist == maxDist && cost < minCost)) {
                            maxDist = dist;
                            minCost = cost;
                            bestCandidate = candidate;
                        }
                    }
                }
            }
        } 
        else {
            // NEGATIVE case: utility function based on coverage and cost
            double bestUtility = -1.0;
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    Cell candidate = cells[i][j];
                    // skip closed cells or visited cells or cells with zero probability
                    if (!candidate.isOpen() || probabilityMap[i][j] == 0.0 || visited[i][j]) continue; 

                    Set<Cell> candidateRegion = new HashSet<>();
                    // sum of the probabilities in cells that the candidate can sense
                    // of which we want to maximize in picking the next sensing center(candidate)
                    double uncoveredProb = 0.0;
                    for (int x = 0; x < dimension; x++) {
                        for (int y = 0; y < dimension; y++) {
                            int manhattan = Math.abs(x - i) + Math.abs(y - j);
                            if (manhattan <= sensoringRange && cells[x][y].isOpen() && probabilityMap[x][y] > 0.0) {
                                Cell target = cells[x][y];
                                if (!covered.contains(target)) {
                                    uncoveredProb += probabilityMap[x][y];
                                    candidateRegion.add(target);
                                }
                            }
                        }
                    }

                    // candidate not valid since all cells it covered has no rat
                    if (uncoveredProb == 0.0) continue;

                    List<Cell> path = findShortestPath(currentCell, candidate);
                    if (path == null || path.size() == 0) continue;
                    int travelCost = path.size();
                    // we want larger sum of probabilities of uncovered cells, and lower travel cost to the target
                    // adding a small constant to avoid division by zero
                    double utility = uncoveredProb / (travelCost+1e-9);
                    if (utility > bestUtility) {
                        bestUtility = utility;
                        bestCandidate = candidate;
                    }
                }
            }
        }

        if (bestCandidate == null) {
            System.err.println("error: no target found");
        }

        return bestCandidate;
    }


    public int getSensoringRange() {
        return sensoringRange;
    }


}
