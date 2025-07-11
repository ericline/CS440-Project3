
public enum Status {
    BLOCKED("Blocked"),
    OPEN("Open"),
    UNKNOWN("UNKNOWN");

    private final String status;

    Status(String status) {
        this.status = status;
    }
}