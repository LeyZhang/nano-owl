import os

# file_path = "../data/data/Person/label/filter_train.txt"

# filtered_lines = []
# with open(file_path, "r") as file:
#     lines = file.readlines()
#     for line in lines:
#         line = line.replace("Person_Det", "Person")
#         filtered_lines.append(line)

# filtered_file_path = "filter_train.txt"
# with open(filtered_file_path, "w") as file:
#     for line in filtered_lines:
#         file.write(line)

filtered_file_path = "filter_train.txt"
gt_file_path = "result.txt"

filtered_lines = []
with open(filtered_file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        filtered_lines.append(line)

gt_lines = []
with open(gt_file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        gt_lines.append(line)

transform_filter_lines = []

for line in filtered_lines:
    check_flag = line.split(" ")[0]
    for target_line in gt_lines:
        target_flag = target_line.split(" ")[0]
        if check_flag == target_flag:
            transform_filter_lines.append(line)
            break

transform_gt_lines = []

for line in gt_lines:
    check_flag = line.split(" ")[0]
    for target_line in transform_filter_lines:
        target_flag = target_line.split(" ")[0]
        if check_flag == target_flag:
            transform_gt_lines.append(line)
            break

transform_filter_file_path = "transform_filter_train.txt"
with open(transform_filter_file_path, "w") as file:
    for line in transform_filter_lines:
        file.write(line + "\n")

transform_gt_path = "transform_result.txt"
with open(transform_gt_path, "w") as file:
    for line in transform_gt_lines:
        file.write(line + "\n")

