import numpy as np
from collections import defaultdict
from PIL import Image

from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)
from torch import nn
import torch

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


# clf = MLP(768, [1024, 2048, 1024, 512, 256, 128], 2)
# clf.load_state_dict(torch.load('model/mlp_new_total_data_best.pth'))
# clf.eval()

current_model = None
predictor = None

engine_paths = {
    "google/owlvit-base-patch32": "../data/owl_image_encoder_base_patch32.engine",
    "google/owlvit-large-patch14": "../data/owl_image_encoder_large_patch14.engine",
    "google/owlv2-base-patch16-ensemble": "../data/owlv2_image_encoder_base_patch16.engine",
    "google/owlv2-large-patch14-ensemble": "../data/owlv2_image_encoder_large_patch14.engine",
}


def load_model(model, engine_path):
    global current_model, predictor
    
    if model != current_model:
        current_model = model

    predictor = OwlPredictor(
        current_model,
        image_encoder_engine=engine_path,
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
    # global clf

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

    output, embeds = predictor.predict(
        image=image, 
        text=text, 
        text_encodings=text_encodings,
        threshold=thresholds,
        nms_threshold=nms_threshold,
        pad_square=False,
        extract=True
    )

    labels = output.labels.cpu().numpy()
    scores = output.scores.detach().cpu().numpy()
    boxes = output.boxes.detach().cpu().numpy()
    indices = output.input_indices.detach().cpu().numpy()

    result = output

    # select_embeds = embeds[:,indices,:].squeeze().detach().cpu().numpy()
    # select_embeds /= (np.linalg.norm(select_embeds, axis=-1, keepdims=True) + 1e-6)
    # pred_labels = clf(torch.from_numpy(select_embeds).float()).detach().cpu().numpy().squeeze()
    # pred_labels = np.argmax(pred_labels, axis=-1)

    # output.labels = labels[pred_labels == 1]
    # output.scores = scores[pred_labels == 1]
    # output.boxes = boxes[pred_labels == 1]



    log = '\n'.join(map(lambda x: str(x).strip('()').replace('array(', '').replace(', dtype=float32', '').replace(' ', ''),
                    zip(labels, list(scores), list(boxes))))

    # output.labels, output.scores, output.boxes = filter_by_conf(labels, scores, boxes, conf_ratio, min_box_size)


    # neg_labels = list(range(len(pos_text), len(text)))
    # output.labels, output.scores, output.boxes = filter_by_neg(output.labels, output.scores, output.boxes, inter_ratio_threshold, neg_labels)

    result_filtered = output
    
    return result, result_filtered, log




def detect_api(image_path, prompt, config):
    input_image = Image.open(image_path)
    pos_prompt = config["prompt"]
    threshold = config["threshold"]
    nms_threshold = config["nms_threshold"]
    conf_ratio = config["conf_ratio"]
    inter_ratio_threshold = config["inter_ratio_threshold"]
    min_box_size = config["min_box_size"]
    neg_prompt = None
    result, result_filtered, log = detect(input_image, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size)
    return result_filtered.labels, result_filtered.scores, result_filtered.boxes

import yaml
import os

if __name__ == "__main__":
    # 读取配置文件
    with open("config/config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    input_folder = config["input_folder"]
    model = config["model"]
    engine_path = config["engine_path"]
    prompt = config["prompt"]
    test_result_file = config["test_result_file"]
    
    # 加载模型
    load_model(model, engine_path)
    
    result_lines = []
    

    for image_name in os.listdir(input_folder):
        # 执行检测并打印结果
        input_image_path = os.path.join(input_folder, image_name)
        labels, scores, boxes = detect_api(input_image_path, prompt, config)
        print(f"result_filtered.labels: {labels}, result_filtered.scores: {scores}, result_filtered.boxes: {boxes}")
    


