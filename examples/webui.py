import numpy as np
import gradio as gr
from collections import defaultdict
import os
from PIL import Image
import torch

from nanoowl.owl_predictor import (
    OwlPredictor,
    OwlEncodeTextOutput,
    OwlDecodeOutput,
)
from nanoowl.fewshot_predictor import (
    FewshotPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.activate0 = nn.GELU()
        for i in range(len(hidden_dims)-1):
            setattr(self, f'fc{i+1}', nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            setattr(self, f'activate{i+1}', nn.GELU())
        self.fcout = nn.Linear(hidden_dims[-1], output_dim)
        self.final_activate = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = self.activate0(x)
        for i in range(len(self.hidden_dims)-1):
            x = getattr(self, f'fc{i+1}')(x)
            x = getattr(self, f'activate{i+1}')(x)
        x = self.fcout(x)
        x = self.final_activate(x)
        return x


clf = MLP(768, [1024, 2048, 1024, 512, 256, 128], 2)
clf.load_state_dict(torch.load('model/mlp_new_total_data_best.pth'))
clf.eval()

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


def detect(image, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, *args):
    global predictor
    fewshot_predictor = FewshotPredictor(
        owl_predictor=predictor
    )


    text = []

    if pos_prompt:    
        pos_prompt = pos_prompt.strip("][()")
        pos_text = pos_prompt.split(',')
        text += pos_text
    else:
        pos_text = []

    if neg_prompt:
        neg_prompt = neg_prompt.strip("][()")
        neg_text = neg_prompt.split(',')
        text += neg_text
    else:
        neg_text = []

    thresholds = threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    
    text_encodings = None
    if any(t for t in text):
        text_encodings = predictor.encode_text(text)

    ref_image_list = []
    ref_prompt_list = []

    for arg in args:
        if type(arg) == list:  # 参考图像
            ref_image_list.append(arg) 
        elif type(arg) == str:        # 参考提示词
            ref_prompt_list.append(arg)

    for i, (ref_images, ref_prompt) in enumerate(zip(ref_image_list, ref_prompt_list)):
        if not ref_images or not ref_prompt:
            continue
        
        ref_embed_list = []
        for ref_image, _ in ref_images:
            # from IPython import embed; embed()
            if ref_image.format == "PNG":
                import io
                ref_image = ref_image.convert("RGB")
                byte_stream = io.BytesIO()
                ref_image.save(byte_stream, 'JPEG')
                byte_stream.seek(0)
                ref_image = Image.open(byte_stream)
            ref_embed = fewshot_predictor.encode_query_image(image=ref_image, text=ref_prompt)
            ref_embed_list.append(ref_embed)
        
        # agregate ref_embed_list
        ref_embed = torch.mean(torch.stack(ref_embed_list), dim=0)
        text_embed = fewshot_predictor.encode_text([ref_prompt]).text_embeds
        ref_embed = torch.mean(torch.stack([ref_embed, text_embed]), dim=0)
        
        if text_encodings:
            text_encodings.text_embeds = torch.cat([text_encodings.text_embeds, ref_embed], dim=0)
        else:
            text_encodings = OwlEncodeTextOutput(text_embeds=ref_embed)

        text.append(ref_prompt + f"_ref_{i}")


    output, embeds = predictor.predict(
        image=image, 
        text=text, 
        text_encodings=text_encodings,
        threshold=thresholds,
        nms_threshold=nms_threshold,
        pad_square=False,
        extract=True
    )


    # labels = output.labels.cpu().numpy()
    # # neg_labels = list(range(len(pos_text), len(pos_text) + len(neg_text)))
    # # neg_scores = output.scores.cpu().numpy()[labels in neg_labels]

    # scores = output.scores.detach().cpu().numpy()
    # boxes = output.boxes.detach().cpu().numpy()
    # indices = output.input_indices.detach().cpu().numpy()
    # select_embeds = embeds[:,indices,:].squeeze().detach().cpu().numpy()

    # select_embeds /= (np.linalg.norm(select_embeds, axis=-1, keepdims=True) + 1e-6)


    # global clf
    # pred_labels = clf(torch.from_numpy(select_embeds).float()).detach().cpu().numpy().squeeze()
    # pred_labels = np.argmax(pred_labels, axis=-1)
    result = draw_owl_output(image, output, text=text, draw_text=True)
    
    # output.labels = labels[pred_labels == 1]
    # output.scores = scores[pred_labels == 1]
    # output.boxes = boxes[pred_labels == 1]

    # log = '\n'.join(map(lambda x: str(x).strip('()').replace('array(', '').replace(', dtype=float32', '').replace(' ', ''),
    #                 zip(labels, list(scores), list(boxes))))

    # output.labels, output.scores, output.boxes = filter_by_conf(labels, scores, boxes, conf_ratio, min_box_size)
    


    # output.labels, output.scores, output.boxes = filter_by_neg(output.labels, output.scores, output.boxes, inter_ratio_threshold, neg_labels)

    result_filtered = draw_owl_output(image, output, text=text, draw_text=True)
    
    return result, result_filtered, log



if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 新大陆物体检测方案")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 输入图像")
                input_img = gr.Image(type="pil", label="Input Image")
                
                gr.Markdown("### 参考图像")
                ref_context = []
                n_tabs = 5
                with gr.Tabs():
                    for i in range(n_tabs):
                        with gr.Tab(label=f"参考{i}"):
                            ref_context.append(gr.Gallery(type="pil", label="参考图像"))
                            ref_context.append(gr.Textbox(label="参考提示词", value=""))

                gr.Markdown("### 模型参数")
                model_choice = gr.Dropdown(list(engine_paths.keys()), label="检测模型")
                pos_prompt = gr.Textbox(label="正向提示词", value="")
                neg_prompt = gr.Textbox(label="反向提示词", value="")
                threshold = gr.Textbox(label="检测阈值", value="0.03")
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
                      inputs=[input_img, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, 
                              *ref_context],
                      outputs=[output_image, output_image_filtered, log])
        
    demo.launch(server_name="0.0.0.0")