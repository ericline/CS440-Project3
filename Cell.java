import java.util.ArrayList;
import java.util.Objects;

public class Cell implements Comparable<Cell>{

    public int dimension; // dimension of the grid
    public int x; // x-coordinate of the cell
    public int y; // y-coordinate of the cell
    public Status status; // marks the status of the cell
    public RatStatus ratStatus;
    public ArrayList<Cell> neighbors; // list of neighboring cells
    public int openNeighbors; // number of open neighbors
    public int costFromStart = Integer.MAX_VALUE; // cost of node from start (for A* algorithm)
    public int totalCost = Integer.MAX_VALUE; // total cost: costFromStart + h(n) (for A* algorithm)
    public Cell prev; // previous visited cell


    public Cell(int x, int y, Status status) {
        this.x = x;
        this.y = y;
        this.status = status;
        this.ratStatus = RatStatus.NoRat;
        this.neighbors = new ArrayList<>();
    }

    public int getX() {
        return x;
    }
    public int getY() {
        return y;
    }

    public Cell getCell() {
        return this;
    }

    public boolean isOpen() {
        return status == Status.OPEN;
    }

    public void setOpen(boolean isOpen) {
        this.status = isOpen ? Status.OPEN : Status.BLOCKED;
        // System.out.println("Cell at (" + x + ", " + y + ") set to " + (isOpen ? "open" : "closed"));
    }

    public void placeRat() {
        this.ratStatus = RatStatus.HasRat;
    }

    public void addNeighbor(Cell cell) {
        neighbors.add(cell);
    }

    public int getOpenNeighbors() {
        openNeighbors = 0;
        for (Cell neighbor : neighbors) {
            if (neighbor.isOpen()) {
                openNeighbors++;
            }
        }
        return openNeighbors;
    }

    public int getClosedNeighbors() {
        int closedNeighbors = 0;
        for (Cell neighbor : neighbors) {
            if (neighbor.status == Status.BLOCKED) {
                closedNeighbors++;
            }
        }
        int n = neighbors.size();
    
        while (n < 4) {
            closedNeighbors++;
            n++;

        }
        return closedNeighbors;
    }


    public boolean blocked() {
        boolean allNeighborsBlockedOrTrapped = true;

        for (Cell neighbor : neighbors) {
            if (neighbor.status != Status.BLOCKED) {
                int blockedAroundNeighbor = 0;
                for (Cell secondLevel : neighbor.neighbors) {
                    if (secondLevel.status == Status.BLOCKED) {
                        blockedAroundNeighbor++;
                    }
                }
                // If this neighbor has at least one open or unknown neighbor, it might be a path
                if (blockedAroundNeighbor < neighbor.neighbors.size()) {
                    allNeighborsBlockedOrTrapped = false;
                    break;
                }
            }
        }
        return allNeighborsBlockedOrTrapped;
    }

    public boolean equalTo(Cell currCell) {
        return this.x == currCell.getX() && this.y == currCell.getY() && this.status == currCell.status;
    }

    @Override
    public int compareTo(Cell other) {
        return Integer.compare(this.totalCost, other.totalCost);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Cell other = (Cell) obj;
        return this.x == other.x && this.y == other.y;
    }


@Override
public int hashCode() {
    return Objects.hash(x, y);
}


}