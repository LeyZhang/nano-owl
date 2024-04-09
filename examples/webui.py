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


def load_model(model):
    global current_model, predictor
    
    if model != current_model:
        current_model = model

    predictor = OwlPredictor(
        current_model,
        image_encoder_engine=engine_paths[current_model],
    )


def inter_ratio(box1, box2):
    x1,y1,x2,y2 = box1 #box1的左上角坐标、右下角坐标
    x3,y3,x4,y4 = box2 #box1的左上角坐标、右下角坐标

    #计算交集的坐标
    x_inter1 = max(x1,x3) #union的左上角x
    y_inter1 = max(y1,y3) #union的左上角y
    x_inter2 = min(x2,x4) #union的右下角x
    y_inter2 = min(y2,y4) #union的右下角y

    # 计算交集部分面积，因为图像是像素点，所以计算图像的长度需要加一
    # 比如有两个像素点(0,0)、(1,0)，那么图像的长度是1-0+1=2，而不是1-0=1
    inter_area = max(0,x_inter2-x_inter1+1)*max(0,y_inter2-y_inter1+1)

    # 分别计算两个box的面积
    area_box1 = (x2-x1+1)*(y2-y1+1)
    area_box2 = (x4-x3+1)*(y4-y3+1)

    #并集面积/box面积
    return inter_area / area_box1, inter_area / area_box2


def filter_by_conf(labels, scores, boxes, conf_ratio, min_box_size):
    labels_filtered = []
    scores_filtered = []
    boxes_filtered = []
    
    label_scores = defaultdict(list)
    for label, score in zip(labels, scores):
        label_scores[label].append(score)

    max_scores = {label: max(scores) for label, scores in label_scores.items()}

    for label, score, box in zip(labels, scores, boxes):
        if score < conf_ratio * max_scores[label]:
            continue

        x1, y1, x2, y2 = box
        if x2 - x1 + 1 < min_box_size or y2 - y1 + 1 < min_box_size:
            continue

        labels_filtered.append(label)
        scores_filtered.append(score)
        boxes_filtered.append(box)

    return labels_filtered, scores_filtered, boxes_filtered


def filter_by_neg(labels, scores, boxes, inter_ratio_threshold, neg_labels):
    neg_boxes = [box for label, box in zip(labels, boxes) if label in neg_labels]
    
    labels_filtered = []
    scores_filtered = []
    boxes_filtered = []

    for label, score, box in zip(labels, scores, boxes):
        if label in neg_labels:
            continue

        inter_ratios = [inter_ratio(box, neg_box)[0] for neg_box in neg_boxes]
        if inter_ratios and max(inter_ratios) > inter_ratio_threshold:
            continue

        labels_filtered.append(label)
        scores_filtered.append(score)
        boxes_filtered.append(box)

    return labels_filtered, scores_filtered, boxes_filtered


def detect(image, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size):
    global predictor

    text = []

    pos_prompt = pos_prompt.strip("][()")
    pos_text = pos_prompt.split(',')
    text += pos_text

    if neg_prompt:
        neg_prompt = neg_prompt.strip("][()")
        neg_text = neg_prompt.split(',')
        text += neg_text

    thresholds = threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    
    text_encodings = predictor.encode_text(text)

    output = predictor.predict(
        image=image, 
        text=text, 
        text_encodings=text_encodings,
        threshold=thresholds,
        nms_threshold=nms_threshold,
        pad_square=False
    )

    labels = output.labels.cpu().numpy()
    scores = output.scores.detach().cpu().numpy()
    boxes = output.boxes.detach().cpu().numpy()

    log = '\n'.join(map(lambda x: str(x).strip('()').replace('array(', '').replace(', dtype=float32', '').replace(' ', ''),
                    zip(labels, list(scores), list(boxes))))

    output.labels, output.scores, output.boxes = filter_by_conf(labels, scores, boxes, conf_ratio, min_box_size)

    result = draw_owl_output(image, output, text=text, draw_text=True)

    neg_labels = list(range(len(pos_text), len(text)))
    output.labels, output.scores, output.boxes = filter_by_neg(output.labels, output.scores, output.boxes, inter_ratio_threshold, neg_labels)

    result_filtered = draw_owl_output(image, output, text=text, draw_text=True)
    
    return result, result_filtered, log


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 新大陆物体检测方案")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 输入图像")
                input_img = gr.Image(type="pil", label="Input Image")
                gr.Markdown("### 模型参数")
                model_choice = gr.Dropdown(list(engine_paths.keys()), label="检测模型")
                pos_prompt = gr.Textbox(label="正向提示词", value="")
                neg_prompt = gr.Textbox(label="反向提示词", value="")
                threshold = gr.Textbox(label="检测阈值", value="0.05")
                nms_threshold = gr.Slider(label="NMS阈值", maximum=1.0, step=0.05, value=0.3)
                gr.Markdown("### 过滤参数")
                conf_ratio = gr.Slider(label="置信度低于最大置信度乘以该参数的将被排除", maximum=1.0, step=0.05, value=0.8)
                inter_ratio_threshold = gr.Slider(label="与反向提示词检测框的交集占比大于该参数的将被排除", maximum=1.0, step=0.05, value=0.5)
                min_box_size = gr.Slider(label="最小检测框大小", maximum=20., step=1, value=10.)
                run_btn = gr.Button("检测")
            with gr.Column(scale=4):
                gr.Markdown("### 运行结果")
                output_image = gr.Image(type="pil", label="检测结果")
                output_image_filtered = gr.Image(type="pil", label="检测结果 (过滤后)")
                log = gr.Textbox(lines=10, label="运行输出")

        model_choice.change(load_model, inputs=model_choice)
        run_btn.click(fn=detect,
                      inputs=[input_img, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size],
                      outputs=[output_image, output_image_filtered, log])
        
    demo.launch(server_name="0.0.0.0")