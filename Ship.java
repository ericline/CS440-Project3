import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Ship{
    private int dimension; 
    private Cell[][] cell;
    int numOfOpenCells = 0;
    static Random random;
    public Cell ratCell = null; 
    protected int nodesProcessed = 0;


    // for cell[a][b] = 0, it is closed; 1 if it is open

    public Ship(int dimension, Status status, int seed) {
        this.dimension = dimension;
        random = new Random(seed);

        this.cell = new Cell[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                this.cell[i][j] = new Cell(i, j, status); 
            }
        }
        assignNeighbors(); // assign neighbors to each cell

    }

    private void assignNeighbors() {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                Cell currCell = cell[i][j];

                // Up
                if (i != 0) currCell.addNeighbor(cell[i - 1][j]);
                // Down
                if (i != dimension - 1) currCell.addNeighbor(cell[i + 1][j]);
                // Left
                if (j != 0) currCell.addNeighbor(cell[i][j - 1]);
                // Right
                if (j != dimension - 1) currCell.addNeighbor(cell[i][j + 1]);
            }
        }
    }

    public static Ship shipGenerator(int dimension, double p, int seed) {
        Ship ship = new Ship(dimension, Status.BLOCKED, seed);

        /*
         * Part I: Randomly open cells until there is no more closed cell with exactly one open neighbor
         */

        // pick a random interior cell to be open 
        Cell inicell = ship.cell[random.nextInt(dimension)][random.nextInt(dimension)];
        inicell.setOpen(true);
        ship.numOfOpenCells++;

        ArrayList<Cell> qualifiedCells = new ArrayList<Cell>();

        // loop to open cells until there is no more closed cell with exactly one open neighbor
        while (true) {
            qualifiedCells.clear();
            // find all the cells that are not open and have exactly one open neighbor
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    // System.out.println("Checking cell at (" + i + ", " + j + ")");
                    if (!ship.cell[i][j].isOpen()) {
                        if (ship.cell[i][j].getNumOfOpenNeighbors() == 1) {
                            qualifiedCells.add(ship.cell[i][j]);
                        }
                    }
                }
            }
            // if there are no more qualified cells, break the while loop
            if (qualifiedCells.isEmpty()) {
                break;
            }
            Cell selectedCell = qualifiedCells.get(random.nextInt(qualifiedCells.size()));
            selectedCell.setOpen(true);
            ship.numOfOpenCells++;
        }

        /*
         * Part II: Reduce the number of dead ends by p
         */

        int iniDeadEndCount = 0;
        ArrayList<Cell> deadEndCells = new ArrayList<>();

        // Generate list of current dead ends
        for(int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                Cell currCell = ship.cell[i][j];
                if (currCell.isOpen() && currCell.getNumOfOpenNeighbors() == 1) {
                    iniDeadEndCount++;
                    deadEndCells.add(currCell);
                }
            }
        }
    // System.out.println("Initial dead end count: " + iniDeadEndCount);
        
        int curDeadEndCount = iniDeadEndCount;
        while(curDeadEndCount > iniDeadEndCount * p){
            // Select random dead end cell from list of dead ends
            Cell selectedCell = deadEndCells.get(random.nextInt(deadEndCells.size()));
            
            // Get neighbors of selected cell
            ArrayList<Cell> nbhs = selectedCell.neighbors;
            
            // Create a list of its blocked neighbors
            List<Cell> closedNeighbors = new ArrayList<>();
            for (Cell neighbor : nbhs) {
                if (neighbor.status == Status.BLOCKED) {
                    closedNeighbors.add(neighbor);
                }
            }
            
            if (!closedNeighbors.isEmpty()) {
                // Select random closed neighbor and open it
                Cell selectedNbh = closedNeighbors.get(random.nextInt(closedNeighbors.size()));
                selectedNbh.setOpen(true);
                ship.numOfOpenCells++;
                curDeadEndCount--;

                // Check if selectedNbh now became a new dead end
                if (selectedNbh.getNumOfOpenNeighbors() == 1) {
                    deadEndCells.add(selectedNbh);
                }

            }
        }

        // System.out.println("Final dead end count: " + curDeadEndCount);
        Cell rat;
        do {
            int r = random.nextInt(dimension);
            int c = random.nextInt(dimension);
            rat = ship.cell[r][c];
        } while (!rat.isOpen());
        rat.placeRat();
        ship.ratCell = rat;  

        return ship;
    }

    // Getter for cell array
    public Cell[][] getCell() {
        return cell;
    }

    // Getter for dimension of ship
    public int getDimension() {
        return dimension;
    }

    public void initializeSpaceRat() {
        // Get list of all open cells
        ArrayList<Cell> openCells = new ArrayList<>();
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if(cell[i][j].isOpen()) {
                    openCells.add(cell[i][j]);
                }
            }
        }
        
        // If there are open cells, randomly set one to have the space rat
        if(!openCells.isEmpty()) {
            openCells.get(random.nextInt(openCells.size())).placeRat();
        }
    }

    public Cell[][] getCells() {
        Cell[][] copy = new Cell[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                copy[i][j] = cell[i][j]; // Create a new Cell
            }
        }
        return copy;
    }


    public Cell getRatCell() {
    return ratCell;
    }

}
