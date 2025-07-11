import javax.swing.*;
import java.awt.*;

public class ShipViewer extends JPanel {
    private final Cell[][] grid;
    private final int cellSize = 20;  // size of each cell in pixels

    public ShipViewer(Cell[][] grid) {
        this.grid = grid;
        int rows = grid.length;
        int cols = grid[0].length;
        setPreferredSize(new Dimension(cols * cellSize, rows * cellSize));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        for (int y = 0; y < grid.length; y++) {
            for (int x = 0; x < grid[0].length; x++) {
                Cell c = grid[x][y];  // x = horizontal, y = vertical

                if (c.status == Status.OPEN) {
                    g.setColor(Color.WHITE);
                } else if (c.status == Status.BLOCKED) {
                    g.setColor(Color.BLACK);
                } else {
                    g.setColor(Color.GRAY);
                }

                g.fillRect(x * cellSize, (grid.length - 1 - y) * cellSize, cellSize, cellSize);
            }
        }

        // Color in grid lines
        g.setColor(Color.DARK_GRAY);
        for (int i = 0; i <= grid.length; i++) {
            g.drawLine(0, i * cellSize, getWidth(), i * cellSize); // horizontal
            g.drawLine(i * cellSize, 0, i * cellSize, getHeight()); // vertical
        }
    }

    public static void show(Cell[][] grid, String windowName) {
        JFrame frame = new JFrame(windowName);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(new ShipViewer(grid));
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
