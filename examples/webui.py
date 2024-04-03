import gradio as gr

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


def detect_objects(image, prompt, threshold, nms_threshold, model, no_roi_align):
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

    result = draw_owl_output(image, output, text=text, draw_text=True)

    return result


# Interface
inputs = [
    gr.Image(type="pil", label="Input Image"),
    gr.Textbox(label="Text Prompt", value="a photo of fire or smoke"),
    gr.Textbox(label="Threshold", value="0.1"),
    gr.Number(label="NMS Threshold", value="0.3"),
    gr.Dropdown(list(engine_paths.keys()), label="Model Choice"),
    gr.Checkbox(label="No ROI Align"),
]

outputs = gr.Image(type="pil", label="Output Image")

gr.Interface(
    fn=detect_objects,
    inputs=inputs,
    outputs=outputs,
    title="Object Detection with Gradio",
).launch()