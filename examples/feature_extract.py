import numpy as np
from collections import defaultdict
import os
from PIL import Image
import torch
import json

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

current_model = None
predictor = None
fewshot_predictor = None

engine_paths = {
    "google/owlvit-base-patch32": "../data/owl_image_encoder_base_patch32.engine",
    "google/owlvit-large-patch14": "../data/owl_image_encoder_large_patch14.engine",
    "google/owlv2-base-patch16-ensemble": "../data/owlv2_image_encoder_base_patch16.engine",
    "google/owlv2-large-patch14-ensemble": "../data/owlv2_image_encoder_large_patch14.engine",
}


def load_model(model, engine_path):
    global current_model, predictor, fewshot_predictor
    
    if model != current_model:
        current_model = model

    predictor = OwlPredictor(
        current_model,
        image_encoder_engine=engine_path,
    )

    fewshot_predictor = FewshotPredictor(
        owl_predictor=predictor
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


def filter_by_conf(labels, scores, boxes, indices, conf_ratio, min_box_size):
    labels_filtered = []
    scores_filtered = []
    boxes_filtered = []
    indices_filtered = []
    
    label_scores = defaultdict(list)
    for label, score in zip(labels, scores):
        label_scores[label].append(score)

    max_scores = {label: max(scores) for label, scores in label_scores.items()}

    for label, score, box, index in zip(labels, scores, boxes, indices):
        if score < conf_ratio * max_scores[label]:
            continue

        x1, y1, x2, y2 = box
        if x2 - x1 + 1 < min_box_size or y2 - y1 + 1 < min_box_size:
            continue

        labels_filtered.append(label)
        scores_filtered.append(score)
        boxes_filtered.append(box)
        indices_filtered.append(index)

    return labels_filtered, scores_filtered, boxes_filtered, indices_filtered


def iou(box1, box2):
    x1,y1,x2,y2 = box1 #box1的左上角坐标、右下角坐标
    x3,y3,x4,y4 = box2 #box1的左上角坐标、右下角坐标

    #计算交集的坐标
    x_inter1 = max(x1,x3) #union的左上角x
    y_inter1 = max(y1,y3) #union的左上角y
    x_inter2 = min(x2,x4) #union的右下角x
    y_inter2 = min(y2,y4) #union的右下角y
    
    inter_area = max(0,x_inter2-x_inter1+1)*max(0,y_inter2-y_inter1+1)

    # 分别计算两个box的面积
    area_box1 = (x2-x1+1)*(y2-y1+1)
    area_box2 = (x4-x3+1)*(y4-y3+1)

    #交集面积/并集面积
    return inter_area / (area_box1 + area_box2 - inter_area)
    

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


idx = 0
def detect(image, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, ref_image_list, ref_prompt_list):
    global fewshot_predictor 

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

    for i, (ref_image, ref_prompt) in enumerate(zip(ref_image_list, ref_prompt_list)):
        if not ref_image or not ref_prompt:
            continue
        ref_embed = fewshot_predictor.encode_query_image(image=ref_image, text=ref_prompt)
        
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

    labels = output.labels.cpu().numpy()
    scores = output.scores.detach().cpu().numpy()
    boxes = output.boxes.detach().cpu().numpy()
    indices = output.input_indices
    


    log = '\n'.join(map(lambda x: str(x).strip('()').replace('array(', '').replace(', dtype=float32', '').replace(' ', ''),
                    zip(labels, list(scores), list(boxes))))

    # output.labels, output.scores, output.boxes, output.input_indices = filter_by_conf(labels, scores, boxes, indices, conf_ratio, min_box_size)

    result = output

    result_filtered = output    
    return result, result_filtered, log, embeds


def generate_embedding_data(ground_truth_lines, image_name, labels, scores, boxes, indices, select_embeddings, iou_threshold = 0.7):
    gtboxes = None
    for line in ground_truth_lines:
        if image_name in line:
            line = line.strip().split(" ")
            image_name = line[0]
            gtboxes = np.array(line[1:], float).reshape(-1, 5)
            break
    if len(gtboxes) == 0: # 背景图片
        embeddings = [select_embedding.tolist() for select_embedding in select_embeddings]
        embeddings = np.array(embeddings)
        return embeddings, np.zeros(len(embeddings))
    total_embeddings = []
    total_ious = []

    for label, score, box, index, select_embedding in zip(labels, scores, boxes, indices, select_embeddings):
        max_iou = 0
        box = box.tolist()
        for gtbox in gtboxes:
            max_iou = max(max_iou, iou(gtbox[:4], box))
        total_embeddings.append(select_embedding.tolist())
        total_ious.append(max_iou)
    total_embeddings = np.array(total_embeddings)
    total_ious = np.array(total_ious)
    return total_embeddings, total_ious
    

def detect_vision_api(image_path, pos_prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, image_ref_paths:list, prompt_ref:list):
    input_image = Image.open(image_path)
    ref_images = [Image.open(image_ref_path) for image_ref_path in image_ref_paths]
    
    result, result_filtered, log, embeds = detect(input_image, pos_prompt, None, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, ref_images, prompt_ref)
    indices = result_filtered.input_indices
    selected_embeds = embeds[:, indices, :].squeeze(0).detach().cpu().numpy()
    return result_filtered.labels, result_filtered.scores, result_filtered.boxes, result_filtered.input_indices, selected_embeds

import yaml
import os
from tqdm import tqdm



if __name__ == "__main__":
    # 读取配置文件
    with open("config/feature_extract.yml", "r") as file:
        config = yaml.safe_load(file)

    # 固定的参数
    conf_ratio = 0
    neg_prompt = None
    inter_ratio_threshold = 1
    min_box_size = 0
    
    # 读取配置文件中的参数
    input_folder = config["input_folder"]    
    output_folder = config["output_folder"]
    ref_image_paths = []
    model = config["model"]
    engine_path = config["engine_path"]
    prompt = config["prompt"]
    threshold   = config["threshold"]
    nms_threshold = config["nms_threshold"]
    iou_threshold = config["iou_threshold"]
    gt_label_file = config["ground_truth_label_file"]  # 标签文件只能用空格分割，格式按照新大陆的格式
    
    # 加载模型
    load_model(model, engine_path)

    target_lines = []
    gt_lines = []
    with open(gt_label_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            gt_lines.append(line)
            line = line.split(" ")[0].split("/")[-1]
            target_lines.append(line)
    print(len(target_lines))
    print(target_lines[:10])

    all_pos_embeddings = []
    all_neg_embeddings = []
    all_embeddings = []
    all_ious = []


    for image_name in tqdm(os.listdir(input_folder)):
        input_image_path = os.path.join(input_folder, image_name)
        if image_name not in target_lines: # 如果不在标签文件中，跳过
            continue
        print(f"=== input image: {image_name} ===")
        # labels, scores, boxes, indices, selected_embeds = detect_vision_api(input_image_path, prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, ref_image_paths, [])

        # 执行检测并打印结果
        # try:
        labels, scores, boxes, indices, selected_embeds = detect_vision_api(input_image_path, prompt, neg_prompt, threshold, nms_threshold, conf_ratio, inter_ratio_threshold, min_box_size, ref_image_paths, [])
        # print(f"indices: {indices}")
        total_embeddings, total_ious = generate_embedding_data(gt_lines, image_name, labels, scores, boxes, indices, selected_embeds, iou_threshold)
        all_embeddings.extend(total_embeddings)
        all_ious.extend(total_ious)
            # from IPython import embed; embed()
        # except Exception as e:
        #     print(f"{image_name} error: {e}")
        #     break
        #     continue
        # print(f"result_filtered.labels: {labels}, result_filtered.scores: {scores}, result_filtered.boxes: {boxes}")
        print("=====================================")
    # save as npz
    # np.savez(os.path.join(output_folder, "pos_image_embeddings.npz"), all_pos_embeddings)
    # np.savez(os.path.join(output_folder, "neg_image_embeddings.npz"), all_neg_embeddings)
    # np.savez(os.path.join(output_folder, "image_embeddings.npz"), embeddings=all_embeddings, ious=all_ious)
    np.savez(os.path.join(output_folder, "image_embeddings.npz"), embeddings=all_embeddings)
    np.savez(os.path.join(output_folder, "all_ious.npz"), ious=all_ious)
