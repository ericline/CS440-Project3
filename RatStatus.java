
public enum RatStatus {
    HasRat("HasRat"),
    Unknown("Unknown"),
    NoRat("NoRat");

    private final String status;

    RatStatus(String status) {
        this.status = status;
    }
}