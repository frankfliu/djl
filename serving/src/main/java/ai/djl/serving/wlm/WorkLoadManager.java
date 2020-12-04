package ai.djl.serving.wlm;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class WorkLoadManager {

    private Executor defaultPredictionExecutor;
    private Map<String, ExecutorService> specialisedPredictionExecutor;

    public WorkLoadManager() {
        specialisedPredictionExecutor = new ConcurrentHashMap<>();
        defaultPredictionExecutor =
                new ThreadPoolExecutor(
                        0, 5, 10L, TimeUnit.SECONDS, new SynchronousQueue<Runnable>());
    }

    public Executor getPredictionExecutor(String modelName) {
        if (specialisedPredictionExecutor.containsKey(modelName)) {
            return specialisedPredictionExecutor.get(modelName);
        } else {
            return defaultPredictionExecutor;
        }
    }

    public void registerSpecialisedExecutorForModel(ModelInfo model) {
        if (specialisedPredictionExecutor.containsKey(model.getModelName())) {
            specialisedPredictionExecutor.get(model.getModelName()).shutdown();
        }
        specialisedPredictionExecutor.put(
                model.getModelName(),
                new ThreadPoolExecutor(
                        model.getMinWorkers(),
                        model.getMaxWorkers(),
                        model.getMaxBatchDelay(),
                        TimeUnit.SECONDS,
                        new SynchronousQueue<Runnable>()));
    }

    public void unregisterSpecialisedExecutorForModel(ModelInfo model) {
        specialisedPredictionExecutor.remove(model.getModelName());
    }
}
