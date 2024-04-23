import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import argparse


def IoU(box, boxes, iou_type='u'):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    if iou_type == 'u':
        ovr = inter / (box_area + area - inter)
    else:
        ovr = inter / np.minimum(box_area, area)
    return ovr


def getpr(allbox, img_gtbox_num, conf, cls_gtbox_num, per_cls_best_score, use_cls):
    allnum = 0
    tp = 0
    fp = 0
    tpy = 0
    back_acc_num = 0
    backnum = len(np.where(np.array(img_gtbox_num) == 0)[0])

    img_tp_num = 0

    all_cls_statics = {}
    if use_cls == 1:
        for i in range(len(cls_gtbox_num)):
            all_cls_statics[str(i)] = {}
            all_cls_statics[str(i)]['record'] = [0, 0, 0]
            all_cls_statics[str(i)]['P'] = [0]
            all_cls_statics[str(i)]['R'] = [0]
    for bid in allbox:
        allnum += 1
        box = allbox[bid]
        img_fp_flag = 0
        if len(box) == 0:
            if img_gtbox_num[allnum-1] == 0:
                back_acc_num += 1
            continue
        box = np.array(box)
        nid = np.where(box[:, 3] > conf)[0]
        box = box[nid]
        if img_gtbox_num[allnum-1] == 0:
            if len(box) == 0:
                back_acc_num += 1
                continue
        img_tp = 0
        if use_cls == 1:
            for _, _, _, f, cls, g, ig in box:
                if g == 0:
                    fp += 1
                    img_fp_flag = 1
                    all_cls_statics[str(int(cls))]['record'][1] += 1
                else:
                    if ig == 1:
                        tpy += 1
                        all_cls_statics[str(int(cls))]['record'][2] += 1
                        img_tp += 1
                    tp += 1
                    all_cls_statics[str(int(cls))]['record'][0] += 1
        else:
            for _, _, _, f, g, ig in box:
                if g == 0:
                    fp += 1
                    img_fp_flag = 1
                else:
                    if ig == 1:
                        tpy += 1
                        img_tp += 1
                    tp += 1

        if img_tp == img_gtbox_num[bid] and img_fp_flag == 0:
            img_tp_num += 1

    if use_cls == 1:
        for i in range(len(all_cls_statics)):
            cls_statics = all_cls_statics[str(i)]['record']
            ctp, cfp, ctpy = cls_statics[0], cls_statics[1], cls_statics[2]
            if ctp + cfp == 0:
                p = 0
            else:
                p = 1 - cfp / (ctp + cfp)
            if cls_gtbox_num[str(i)][0] == 0:
                r = 0
            else:
                r = ctpy / cls_gtbox_num[str(i)][0]
            if p + r == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            ff = 1 - p
            lou = 1 - r
            all_cls_statics[str(i)]['P'] = p
            all_cls_statics[str(i)]['R'] = r
            all_cls_statics[str(i)]['F1'] = f1
            all_cls_statics[str(i)]['ff'] = ff
            all_cls_statics[str(i)]['lou'] = lou

            ps = 40 - max(0, 70 - p * 100)
            ws = 30 - max(0, ff * 100 - 20)
            ls = 30 - max(0, lou * 100 - 10)
            if ps < 0:
                ps = 0
            if ws < 0:
                ws = 0
            if ls < 0:
                ls = 0
            score = ps + ws + ls

            if score > per_cls_best_score[str(i)][13]:
                per_cls_best_score[str(i)] = [round(conf, 2),
                                              cls_gtbox_num[str(i)][0],
                                              all_cls_statics[str(i)]['record'][0],
                                              all_cls_statics[str(i)]['record'][2],
                                              all_cls_statics[str(i)]['record'][1],
                                              round(p, 4),
                                              round(ff, 4),
                                              round(lou, 4),
                                              round(r, 4),
                                              round(ps, 2),
                                              round(ws, 2),
                                              round(ls, 2),
                                              round(f1, 2),
                                              round(score, 2)]

    back_result = 0
    if backnum == 0:
        back_result = 1
    else:
        back_result = back_acc_num/backnum
    return allnum, tp, fp, tpy, back_result, per_cls_best_score, img_tp_num


def show(x_list, y_list):
    ax = plt.gca()
    ax.scatter(x_list, y_list, c='r', s=2, alpha=0.5)
    ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)
    plt.show()


def getresult(dtpath, gtpath, dtsplit=';', gtsplit=';', use_cls=0, cls_num=1, iou_th=0.5, iou_type='u', write_csv=0, ingore_size=10):
    alldata = open(dtpath, 'r').read().splitlines()
    allgt = open(gtpath, 'r').read().splitlines()

    label = {}
    gt_label = []
    dt_label = []
    name_label = []
    idnum = len(alldata)
    gtnum = 0
    img_gtbox_num = []

    if len(alldata) != len(allgt):
        print('dt num is', len(alldata), 'does not match the number of gt which is', len(allgt))
        return name_label, gt_label, dt_label, label

    # match dt and gt order
    dic = {}
    new_all_data = []
    for i in range(len(alldata)):
        data_i = alldata[i]
        img_name_i = data_i.strip().split('.jpg')[0]
        dic[img_name_i] = data_i
    for i in range(len(allgt)):
        image_name = allgt[i].strip().split('.jpg')[0]
        if image_name not in dic:
            print(image_name, 'is not in dt.txt')
            return name_label, gt_label, dt_label, label
        line = dic[image_name]
        new_all_data.append(line)
    alldata = new_all_data

    if use_cls == 0:
        cls_num = 1

    cls_gtbox_num = {}
    per_cls_best_score = {}
    for i in range(cls_num):
        cls_gtbox_num[str(i)] = [0]
        per_cls_best_score[str(i)] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(idnum):
        data = alldata[i]

        data = data.strip().split(dtsplit)
        name_label.append(data[0])

        gt = allgt[i]
        gt = gt.strip().split(gtsplit)

        box = np.array(data[1:], float).reshape(-1, 6)
        gtbox = np.array(gt[1:], float).reshape(-1, 5)

        if use_cls == 0:
            box = box[:, :-1]

        dt_label.append(box)
        gt_label.append(gtbox)
        label[i] = []
        ignore_id = []

        for l in range(len(gtbox)):
            bb = gtbox[l]
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            if w < ingore_size or h < ingore_size:
                ignore_id.append(l)
            else:
                if use_cls == 1:
                    cls_gtbox_num[str(int(bb[4]))][0] += 1
        gtnum = gtnum + len(gtbox) - len(ignore_id)
        img_gtbox_num.append(len(gtbox) - len(ignore_id))
        if len(gtbox) == 0:
            bnum = len(box)
            for boxid in range(bnum):
                bb = box[boxid]
                if use_cls == 1:
                    label[i].append([i, -1, boxid, bb[-2], bb[-1], 0, 1])
                else:
                    label[i].append([i, -1, boxid, bb[-1], 0, 1])
            continue
        ex_id = []
        bnum = len(box)
        for boxid in range(bnum):
            bb = box[boxid]
            if bb[2] - bb[0] < ingore_size or bb[3] - bb[1] < ingore_size:
                continue
            iou = IoU(bb, gtbox, iou_type)
            maxiou = np.max(iou)
            if maxiou >= iou_th:
                bid = np.argmax(iou)
                if use_cls == 1:
                    if bb[-1] == gtbox[bid][-1]:
                        if bid in ignore_id:
                            label[i].append([i, bid, boxid, bb[-2], bb[-1], 1, -1])
                        else:
                            if bid not in ex_id:
                                ex_id.append(bid)
                                label[i].append([i, bid, boxid, bb[-2], bb[-1], 1, 1])
                            else:
                                label[i].append([i, bid, boxid, bb[-2], bb[-1], 1, -1])
                    else:
                        label[i].append([i, bid, boxid, bb[-2], bb[-1], 0, 1])
                else:
                    if bid in ignore_id:
                        label[i].append([i, bid, boxid, bb[-1], 1, -1])
                    else:
                        if bid not in ex_id:
                            ex_id.append(bid)
                            label[i].append([i, bid, boxid, bb[-1], 1, 1])
                        else:
                            label[i].append([i, bid, boxid, bb[-1], 1, -1])
            else:
                if use_cls == 1:
                    label[i].append([i, -1, boxid, bb[-2], bb[-1], 0, 1])
                else:
                    label[i].append([i, -1, boxid, bb[-1], 0, 1])
    allresult = []
    for con in np.arange(0.01, 0.95, 0.01):
        allnum, tp, fp, tpy, back_acc, per_cls_best_score, img_tp = getpr(label, img_gtbox_num, con, cls_gtbox_num,
                                                                          per_cls_best_score, use_cls)
        if (tp + fp) == 0:
            continue
        p = 1 - fp / (tp + fp)
        r = tpy / gtnum
        # f1 = 2 * p * r / (p + r)
        ff = 1 - p
        lou = 1 - r
        acc = tp / (tp + lou * gtnum + ff * (tp + fp))
        img_acc = img_tp / allnum
        ps = max(0, 40 - max(0, 80 - p * 100))
        ws = max(0, 30 - max(0, ff * 100 - 20))
        ls = max(0, 30 - max(0, lou * 100 - 10))
        if ps < 0:
            ps = 0
        if ws < 0:
            ws = 0
        if ls < 0:
            ls = 0
        score = ps + ws + ls

        allresult.append([dtpath, con, gtnum, tp, tpy, fp, p, ff, lou, r, acc, img_acc, back_acc, ps, ws, ls, score])
        print("conf: ", round(con, 2), " P: ", p, " R: ", r, "ACC:", acc, "IMG_ACC:", img_acc, "BACK_ACC:", back_acc,
              "score:", score)

    allresult = np.array(allresult)
    score_maxid = np.argmax(allresult[:, -1].astype(float))
    img_acc_maxid = np.argmax(allresult[:, -5].astype(float))
    acc_maxid = np.argmax(allresult[:, -6].astype(float))
    if write_csv:
        best_score_csvfile = open("best_threshold_score.csv", "a")
        best_score_writer = csv.writer(best_score_csvfile)
        best_score_writer.writerow(allresult[score_maxid])

        best_acc_csvfile = open("best_threshold_acc.csv", "a")
        best_acc_writer = csv.writer(best_acc_csvfile)
        best_acc_writer.writerow(allresult[acc_maxid])

        best_im_acc_csvfile = open("best_threshold_im_acc.csv", "a")
        best_im_acc_writer = csv.writer(best_im_acc_csvfile)
        best_im_acc_writer.writerow(allresult[img_acc_maxid])
    for i in range(len(allresult)):
        print(allresult[i])
        if write_csv:
            all_csvfile = open("all_threshold_score.csv", "a")
            all_writer = csv.writer(all_csvfile)
            all_writer.writerow(allresult[i])
            
    if use_cls == 1:
        for i in range(len(per_cls_best_score)):
            print(str(i), " ", per_cls_best_score[str(i)])
            if write_csv and use_cls == 1:
                cls_csvfile = open("cls_threshold_score.csv", "a")
                cls_writer = csv.writer(cls_csvfile)
                cls_writer.writerow([str(i)] + per_cls_best_score[str(i)])
    print("Best score: ", allresult[score_maxid])
    print("Best Acc: ", allresult[acc_maxid])
    print("Best Image Acc: ", allresult[img_acc_maxid])

    if write_csv:
        best_score_csvfile.close()
        best_acc_csvfile.close()
        best_im_acc_csvfile.close()
        all_csvfile.close()
        if use_cls == 1:
            cls_csvfile.close()

    return name_label, gt_label, dt_label, label


# show(allresult[:,-2],allresult[:,-1])


def draw_false(name_label, gt_label, dt_label, label, conf):
    for bid in label:
        box = label[bid]
        if len(box) == 0:
            continue
        box = np.array(box)
        nid = box[:, 3] > conf
        box = box[nid]
        box = box[box[:, -2] == 0]
        if len(box) == 0:
            continue
        else:
            img = cv2.imread(name_label[bid])
            nn = name_label[bid].strip().split('/')[-1]
            for _, gid, boxid, _, _, _ in box:
                gtbox = gt_label[bid]
                # print(boxid, dt_label[bid].shape)
                dtbox = dt_label[bid][int(boxid)]
                for gt in gtbox:
                    # print(gtbox)
                    cv2.putText(img, str(int(gt[4])), (int(gt[0]), int(gt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 1)

                    cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 255, 0), 3)
                cv2.putText(img, str(int(dtbox[4])), (int(dtbox[0]), int(dtbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 255), 1)
                cv2.rectangle(img, (int(dtbox[0]), int(dtbox[1])), (int(dtbox[2]), int(dtbox[3])), (0, 255, 255), 3)
            cv2.imwrite('false/' + nn, img)


def parse_args():
    parser = argparse.ArgumentParser(description='ODT Training')
    parser.add_argument('--dtpath', default='/home/linyuheng/NL_VisStore/trunk/sha/NL_VD_Service/build/result16_2.txt')
    parser.add_argument('--dtsplit', default=';', type=str, help='data split')
    parser.add_argument('--gtpath', default='/home/linyuheng/行人新增数据20220520/行人测试集20221012/masked_data/行人测试集2.txt')
    parser.add_argument('--gtsplit', default=' ', type=str, help='data split')
    parser.add_argument('--use_cls', default=1, type=int, help='use class')
    parser.add_argument('--cls_num', default=1, type=int, help='number of classes')
    parser.add_argument('--iou_th', default=0.5, type=float, help='IoU threshold')
    parser.add_argument('--iou_type', default='u', type=str, help='IoU type')
    parser.add_argument('--ignore_size', default=20, type=int, help='ignore box size')
    parser.add_argument('--write_csv', default=0, type=int, help='write csv flag')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    name_label, gt_label, dt_label, label = getresult(args.dtpath, args.gtpath, args.dtsplit, args.gtsplit, args.use_cls, args.cls_num, args.iou_th, args.iou_type, args.write_csv, args.ignore_size)

