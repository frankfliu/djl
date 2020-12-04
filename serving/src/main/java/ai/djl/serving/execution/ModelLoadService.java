package ai.djl.serving.execution;

import ai.djl.modality.Input;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.ModelManager;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelLoadService {

    private static final Logger logger = LoggerFactory.getLogger(ModelLoadService.class);

    public ModelLoadService() {}

    public CompletableFuture<ModelInfo> getOrLoadModel(Input input, String modelName)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ModelInfo model = modelManager.getModels().get(modelName);
        if (model != null) {
            return CompletableFuture.completedFuture(model);
        } else {
            String regex = ConfigManager.getInstance().getModelUrlPattern();
            if (regex == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            String modelUrl = input.getProperty("model_url", null);
            if (modelUrl == null) {
                byte[] buf = input.getContent().get("model_url");
                if (buf == null) {
                    throw new ModelNotFoundException("Parameter model_url is required.");
                }
                modelUrl = new String(buf, StandardCharsets.UTF_8);
                if (!modelUrl.matches(regex)) {
                    throw new ModelNotFoundException("Permission denied: " + modelUrl);
                }
            }

            logger.info("Loading model {} from: {}", modelName, modelUrl);

            return modelManager.registerModel(modelName, modelUrl, 1, 0);
        }
    }
}
