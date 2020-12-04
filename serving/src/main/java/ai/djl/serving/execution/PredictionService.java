package ai.djl.serving.execution;

import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.translate.TranslateException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);

    private WorkLoadManager executorProvider;
    private ModelLoadService modelLoadService;

    private Executor modelLoadExecutor = Executors.newSingleThreadExecutor();

    public PredictionService(WorkLoadManager executorProvider, ModelLoadService modelLoadService) {
        this.executorProvider = executorProvider;
        this.modelLoadService = modelLoadService;
    }

    public CompletableFuture<Output> predict(Input input, String modelName)
            throws ModelNotFoundException {
        return modelLoadService
                .getOrLoadModel(input, modelName)
                .thenApply(model -> model.queued())
                .thenApplyAsync(
                        model -> {
                            model.running();
                            Predictor predictor = model.getModel().newPredictor();
                            try {

                                return (Output) predictor.predict(input);

                            } catch (TranslateException e) {
                                predictor.close();
                                throw new PredictionExecutionException(e);
                            } finally {
                                predictor.close();
                                model.finished();
                            }
                        },
                        executorProvider.getPredictionExecutor(modelName));
    }
}
