import os
import matplotlib.pyplot as plt
import numpy as np

# List of classes
type45 = "i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
classes = type45.split(",")

# Directories
wd = os.getcwd()
yolo_labels_dir = os.path.join(wd, "VOCdevkit/VOC2007/YOLOLabels")
image_dir = os.path.join(wd, "VOCdevkit/VOC2007/JPEGImages")


def get_all_files(path, extension):
    return [f for f in os.listdir(path) if f.endswith(extension)]


def check_dataset():
    image_files = get_all_files(image_dir, ".jpg")
    label_files = get_all_files(yolo_labels_dir, ".txt")

    # Initialize counters and data structures
    class_image_count = {cls: 0 for cls in classes}
    class_bbox_count = {cls: 0 for cls in classes}
    total_images = len(image_files)
    total_bboxes = 0
    missing_labels = []
    missing_images = []
    class_not_in_45 = []

    for image_file in image_files:
        label_file = image_file.replace(".jpg", ".txt")
        if label_file not in label_files:
            missing_labels.append(image_file)

    for label_file in label_files:
        image_file = label_file.replace(".txt", ".jpg")
        if image_file not in image_files:
            missing_images.append(label_file)

        with open(os.path.join(yolo_labels_dir, label_file), "r") as f:
            bboxes = f.readlines()
            total_bboxes += len(bboxes)
            for bbox in bboxes:
                cls_id = int(bbox.split()[0])
                cls_name = classes[cls_id]
                if cls_name in classes:
                    class_bbox_count[cls_name] += 1
                else:
                    class_not_in_45.append(cls_name)

    # Count images per class
    for label_file in label_files:
        with open(os.path.join(yolo_labels_dir, label_file), "r") as f:
            bboxes = f.readlines()
            counted_classes = set()
            for bbox in bboxes:
                cls_id = int(bbox.split()[0])
                cls_name = classes[cls_id]
                if cls_name in classes and cls_name not in counted_classes:
                    class_image_count[cls_name] += 1
                    counted_classes.add(cls_name)

    avg_bboxes_per_image = total_bboxes / total_images if total_images > 0 else 0

    print("Total number of images:", total_images)
    print("Total number of bounding boxes:", total_bboxes)
    print("Average number of bounding boxes per image:", avg_bboxes_per_image)
    print("Missing label files for images:", missing_labels)
    print("Missing image files for labels:", missing_images)
    print("Classes not in the specified 45 classes:", list(set(class_not_in_45)))

    # Plotting
    plt.figure(figsize=(15, 7))

    # Plotting image count per class
    plt.subplot(1, 2, 1)
    plt.bar(class_image_count.keys(), class_image_count.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Class")

    # Plotting bbox count per class
    plt.subplot(1, 2, 2)
    plt.bar(class_bbox_count.keys(), class_bbox_count.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Number of Bounding Boxes")
    plt.title("Number of Bounding Boxes per Class")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_dataset()
