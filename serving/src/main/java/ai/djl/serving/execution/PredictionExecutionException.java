package ai.djl.serving.execution;

public class PredictionExecutionException extends RuntimeException {

    public PredictionExecutionException() {}

    public PredictionExecutionException(String message) {
        super(message);
    }

    public PredictionExecutionException(Throwable cause) {
        super(cause);
    }

    public PredictionExecutionException(String message, Throwable cause) {
        super(message, cause);
    }

    public PredictionExecutionException(
            String message,
            Throwable cause,
            boolean enableSuppression,
            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
