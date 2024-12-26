package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.ZeroShotClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.translator.ZeroShotClassificationInput;
import ai.djl.modality.nlp.translator.ZeroShotClassificationOutput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ZeroShotClassification {

    private static final Logger logger = LoggerFactory.getLogger(ZeroShotClassification.class);

    private ZeroShotClassification() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        ZeroShotClassificationOutput ret = predict(false);
        logger.info("{}", JsonUtils.GSON_PRETTY.toJson(ret));

        ret = predict(true);
        logger.info("{}", JsonUtils.GSON_PRETTY.toJson(ret));
    }

    public static ZeroShotClassificationOutput predict(boolean multiLabels)
            throws ModelException, IOException, TranslateException {
        Path path =
                Paths.get(
                        "/Users/frankliu/source/junkyard/ptest/huggingface/zero-shot-classification/models/model.pt");

        Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> criteria =
                Criteria.builder()
                        .setTypes(
                                ZeroShotClassificationInput.class,
                                ZeroShotClassificationOutput.class)
                        .optModelPath(path)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new ZeroShotClassificationTranslatorFactory())
                        .build();
        String prompt = "one day I will see the world";
        String[] candidates = {"travel", "cooking", "dancing", "exploration"};

        try (ZooModel<ZeroShotClassificationInput, ZeroShotClassificationOutput> model =
                        criteria.loadModel();
                Predictor<ZeroShotClassificationInput, ZeroShotClassificationOutput> predictor =
                        model.newPredictor()) {
            ZeroShotClassificationInput input =
                    new ZeroShotClassificationInput(prompt, candidates, multiLabels);
            return predictor.predict(input);
        }
    }
}
