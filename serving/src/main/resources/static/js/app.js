$('#modelApp').change(function() {
    var app = $(this).val();
    var select = $('#modelApp2');
    select.empty();
    select.append("<option value=''>Choose...</option>");
    var inputClass = $('#inputClass');

    if (app == "CV") {
        select.append("<option value='cv/image_classification'>Image Classification</option>");
        select.append("<option value='cv/object_detection'>Object Detection</option>");
        select.append("<option value='cv/semantic_segmentation'>Semantic Segmentation</option>");
        select.append("<option value='cv/instance_segmentation'>Instance Segmentation</option>");
        select.append("<option value='cv/pose_estimation'>Pose Estimation</option>");
        select.append("<option value='cv/action_recognition'>Action Recognition</option>");
        select.append("<option value='cv/*'>Other</option>");
        inputClass.val("ai.djl.modality.cv.Image");
    } else if (app == "NLP") {
        select.append("<option value='nlp/'>Question and Answer</option>");
        select.append("<option value='nlp/'>Text Classification</option>");
        select.append("<option value='nlp/'>Sentiment Analysis</option>");
        select.append("<option value='nlp/'>Word Embedding</option>");
        select.append("<option value='nlp/'>Machine Translation</option>");
        select.append("<option value='nlp/'>Multiple Choice</option>");
        select.append("<option value='nlp/'>Other</option>");
    } else if (app == "Tabular") {
        select.append("<option value='tabular/linear_regression'>Linear Regression</option>");
        select.append("<option value='tabular/*'>Other</option>");
    }
});

$('#modelApp2').change(function() {
    var optionSelected = $(this).find("option:selected");
    var app = optionSelected.text();
    var outputClass = $('#outputClass');

    if (app == "Image Classification") {
        outputClass.val("ai.djl.modality.Classifications");
    } else if (app == "Object Detection") {
        outputClass.val("ai.djl.modality.cv.output.DetectedObjects");
    } else {
        outputClass.val("");
    }
});
