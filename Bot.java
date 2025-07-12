
import java.util.Random;
import java.util.Set;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;

public class Bot {
    public int x, y;     
    public Cell[][] map;            // Map of ship
    public Ship ship;               // Original ship
    public int dimension;           // Dimension of ship
    public int movesTaken = 0;      // Track number of moves taken
    public double[][][][] T; // T[bx][by][rx][ry] = expected moves to catch rat
    public double[][][][] optimalAction;


    // T(b_x, b_y, r_x, r_y) = min over actions a [1 + Σ P(next_state | current_state, a) * T(next_state)]

    public int getX() {
     return x; 
    }
    public int getY() {
     return y; 
    }
    public int getDimension() {
     return dimension; 
    } 

    public Bot(Ship ship, int seed) {
        this.ship = ship;
        this.map = ship.getCell();
        this.dimension = ship.getDimension();
        this.T = new double[dimension][dimension][dimension][dimension];


        // Set x and y to a random open cell in ship's map.
        getStartingPoint(seed);

        // Initialize T array, setting all values to 
        initializeT();

    }

    private void initializeT() {
        // Initialize all values
        for (int bx = 0; bx < dimension; bx++) {
            for (int by = 0; by < dimension; by++) {
                for (int rx = 0; rx < dimension; rx++) {
                    for (int ry = 0; ry < dimension; ry++) {
                        if (!map[bx][by].isOpen() || !map[rx][ry].isOpen()) {
                            // Invalid States
                            T[bx][by][rx][ry] = Double.POSITIVE_INFINITY;
                        } else if (bx == rx && by == ry) {
                            // Bot and rat in same cell
                            T[bx][by][rx][ry] = 0.0;
                        } else {
                            T[bx][by][rx][ry] = Double.POSITIVE_INFINITY;
                        }
                    }
                }
            }
        }
    }

    public boolean move(int new_X, int new_Y) {
        // If the moved spot is within the ship's bounds
        if (inBounds(new_X, new_Y)) {
            // If the moved spot is open to move into, update location
            if (map[new_X][new_Y].isOpen()) {
                x = new_X;
                y = new_Y;
                movesTaken++;
                return true;
            }
        }
        // Else no move
        return false;
    }

    // A* Algorithm with Manhattan Distance Heuristic
    public List<Cell> findShortestPath(Cell target) {

        resetCosts();

        PriorityQueue<Cell> fringe = new PriorityQueue<>(Comparator.comparingInt(c -> c.totalCost));
        Set<Cell> closedSet = new HashSet<>();

        Cell startPoint = map[x][y];

        startPoint.costFromStart = 0;
        startPoint.totalCost = manhattan_distance(startPoint, target);
        fringe.add(startPoint); // Add starting cell to fringe
        
        while (!fringe.isEmpty()) {
            Cell curr = fringe.poll();

            // If target is reached, stop and return path.
            if (curr.equals(target)) {
                return reconstructPath(curr);
            }

            closedSet.add(curr);

            // Explore all open and in bound neighbors
            for (Cell neighbor : curr.neighbors) {
                if (neighbor.isOpen() && inBounds(neighbor.x, neighbor.y) && !closedSet.contains(neighbor)) {
                    int temp_dist = curr.costFromStart + 1;

                    if (temp_dist < neighbor.costFromStart) {
                        neighbor.costFromStart = temp_dist;
                        neighbor.totalCost = temp_dist + manhattan_distance(neighbor, target);
                        neighbor.prev = curr;
                        fringe.add(neighbor);
                    }
                }
            }
        }
        // No path found or error.
        System.out.println("No path found");
        return new ArrayList<>();
    }


    private void resetCosts() {
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map.length; j++) {
                Cell cell = map[i][j];
                cell.costFromStart = Integer.MAX_VALUE;
                cell.totalCost = Integer.MAX_VALUE;
                cell.prev = null;
            }
        }
    }

    private List<Cell> reconstructPath(Cell target) {
        List<Cell> path = new ArrayList<>();
        for (Cell cell = target; cell != null; cell = cell.prev) {
            path.add(cell);
        }
        Collections.reverse(path);
        return path;
    }

    public void followPath(List<Cell> path) {
        for (int i = 1; i < path.size(); i++) {
            Cell nextCell = path.get(i);
            boolean success = move(nextCell.x, nextCell.y);

            // Sanitation Check by Eric, remove later if I remember
            if (!success) {
                System.out.println("Move Error at (" + nextCell.x + ", " + nextCell.y + ")");
            }
        }
        
    }

    // Checks coordinates are within the map
    protected boolean inBounds(int x, int y) {
        return (x >= 0 && x < dimension) && (y >= 0 && y < dimension);
    }

    // Obtain random starting point for bot
    public void getStartingPoint(int seed) {

        Random rand = new Random(seed);
        while (true) {
            int row = rand.nextInt(dimension);
            int col = rand.nextInt(dimension);
            if (map[row][col].isOpen()) {
                this.x = row;
                this.y = col;
                return;
            }
        }
    }

    // Obtain furthest unknown cell in internal map
    public Cell getFurthestCell(Cell start) {
        Cell furthest = null;
        int maxDist = -1;

        for (int row = 0; row < map.length; row++) {
            for (int col = 0; col < map[0].length; col++) {
                Cell candidate = map[row][col];

                if (candidate.status == Status.OPEN) {
                    int dist = manhattan_distance(start, candidate);
                    if (dist > maxDist) {
                        maxDist = dist;
                        furthest = candidate;
                    }
                }
            }
        }

        return furthest;
    }

    // Manhattan Heuristic
    public int manhattan_distance(Cell a, Cell b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    // Prints out representation of bot's internal map
    public void printInternalMap(Cell[][] internalGrid) {
        for (int y = 0; y < internalGrid[0].length; y++) {
            for (int x = 0; x < internalGrid.length; x++) {
                Cell cell = internalGrid[x][y];
                switch (cell.status) {
                    case UNKNOWN:
                        System.out.print("U ");
                        break;
                    case BLOCKED:
                        System.out.print("X ");
                        break;
                    case OPEN:
                        System.out.print("O ");
                        break;
                }
            }
            System.out.println();
        }
    }

    public int getNumberOfMoves() {
        return movesTaken;
    }

}