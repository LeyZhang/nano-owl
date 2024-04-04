import numpy as np
import gradio as gr
from collections import defaultdict

from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)


current_model = None
predictor = None

engine_paths = {
    "google/owlvit-base-patch32": "../data/owl_image_encoder_base_patch32.engine",
    "google/owlvit-large-patch14": "../data/owl_image_encoder_large_patch14.engine",
    "google/owlv2-base-patch16-ensemble": "../data/owlv2_image_encoder_base_patch16.engine",
    "google/owlv2-large-patch14-ensemble": "../data/owlv2_image_encoder_large_patch14.engine",
}


def detect_objects(image, prompt, threshold, filter_ratio, nms_threshold, model, no_roi_align):
    global current_model, predictor

    prompt = prompt.strip("][()")
    text = prompt.split(',')

    thresholds = threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]

    if model != current_model:
        current_model = model
        print(engine_paths[current_model])

        predictor = OwlPredictor(
            current_model,
            image_encoder_engine=engine_paths[current_model],
        )
    
    predictor.no_roi_align = no_roi_align

    text_encodings = predictor.encode_text(text)

    output = predictor.predict(
        image=image, 
        text=text, 
        text_encodings=text_encodings,
        threshold=thresholds,
        nms_threshold=float(nms_threshold),
        pad_square=False
    )

    labels = output.labels.cpu().numpy()
    scores = output.scores.detach().cpu().numpy()
    boxes = output.boxes.detach().cpu().numpy()

    label_scores = defaultdict(list)
    for label, score in zip(labels, scores):
        label_scores[label].append(score)

    max_scores = {label: max(scores) for label, scores in label_scores.items()}

    to_remove = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if score < filter_ratio * max_scores[label]:
            to_remove.append(index)

    output.scores = [score for index, score in enumerate(scores) if index not in to_remove]
    output.labels = [label for index, label in enumerate(labels) if index not in to_remove]
    output.boxes = [box for index, box in enumerate(boxes) if index not in to_remove]

    result = draw_owl_output(image, output, text=text, draw_text=True)

    return result, '\n'.join(map(lambda x: str(x).strip('()').replace('array(', '').replace(', dtype=float32', ''), zip(output.labels, list(output.scores), list(output.boxes))))


inputs = [
    gr.Image(type="pil", label="Input Image"),
    gr.Textbox(label="Text Prompt", value="a photo of fire or smoke"),
    gr.Textbox(label="Threshold", value="0.05"),
    gr.Slider(label="Filter Ratio", maximum=1.0, step=0.05, value=0.8),
    gr.Slider(label="NMS Threshold", maximum=1.0, step=0.05, value=0.3),
    gr.Dropdown(list(engine_paths.keys()), label="Model Choice"),
    gr.Checkbox(label="No ROI Align"),
]

outputs = [
    gr.Image(type="pil", label="Output Image"),
    gr.Textbox(lines=10, label="Output")
]

gr.Interface(
    fn=detect_objects,
    inputs=inputs,
    outputs=outputs,
    title="Object Detection with Gradio",
).launch()