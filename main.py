import os
import json

import torch
from torchvision.io import write_png
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from util import BoxUtil, AverageMeter
from data.dataset import get_dataloaders
from models import OwlViT, FocalBoxLoss, PostProcess


def coco_to_model_input(boxes, metadata):
    boxes = BoxUtil.box_convert(boxes, "xywh", "xyxy")
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], mode="down"
    )

    return boxes


def model_output_to_image(boxes, metadata):
    # Model outputs in xyxy normalized coordinates, so scale up
    # before overlaying on image
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], mode="up"
    )

    return boxes


def reverse_labelmap(labelmap):
    return {
        v["new_idx"]: {"actual_category": k, "name": v["name"]}
        for k, v in labelmap.items()
    }


def invalid_batch(boxes):
    # Some images don't have box annotations. Just skip these
    return boxes.size(1) == 0


if __name__ == "__main__":
    n_epochs = 50
    save_train_debug_boxes = False

    train_dataloader, test_dataloader, train_labelcounts = get_dataloaders()
    train_labelcounts = [train_labelcounts[i] for i in sorted(train_labelcounts)]
    labelmap = train_dataloader.dataset.labelmap
    classmap = reverse_labelmap(labelmap)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OwlViT(num_classes=len(labelmap)).to(device)
    criterion = FocalBoxLoss(device, train_labelcounts)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    postprocess = PostProcess()

    model.train()
    for epoch in range(n_epochs):
        if save_train_debug_boxes:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        cls_loss = AverageMeter()
        box_loss = AverageMeter()

        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
        ):
            if invalid_batch(boxes):
                continue

            model.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Predict
            all_pred_boxes, pred_classes = model(image)

            _box_loss, _cls_loss = criterion(
                all_pred_boxes, pred_classes, boxes, labels
            )

            loss = _box_loss + _cls_loss
            loss.backward()
            optimizer.step()

            box_loss.update(_box_loss)
            cls_loss.update(_cls_loss)

        print(box_loss.get_value(), "\t", cls_loss.get_value())
        box_loss.reset()
        cls_loss.reset()

    model.eval()
    os.makedirs("eval/results", exist_ok=True)
    results = []
    with torch.no_grad():
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(test_dataloader, ncols=60)
        ):
            if invalid_batch(boxes):
                continue

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Get predictions and save output
            pred_boxes, pred_classes, scores = postprocess(*model(image))
            pred_boxes = pred_boxes.cpu()

            pred_classes_with_names = []
            for _pcwn in pred_classes:
                pred_classes_with_names.append(
                    [classmap[_pred_class.item()]["name"] for _pred_class in _pcwn]
                )

            pred_boxes = model_output_to_image(pred_boxes, metadata)
            image_with_boxes = BoxUtil.draw_box_on_image(
                metadata["impath"].pop(), pred_boxes, pred_classes_with_names
            )
            write_png(image_with_boxes, f"eval/{i}.jpg")

            # Write in coco format
            pred_boxes = BoxUtil.box_convert(pred_boxes, "xyxy", "xywh")

            for _pred_boxes, _pred_classes, _scores in zip(
                pred_boxes, pred_classes, scores
            ):
                for _pred_box, _pred_class, _score in zip(
                    _pred_boxes.tolist(), _pred_classes.tolist(), _scores.tolist()
                ):
                    results.append(
                        {
                            "image_id": metadata["image_id"].item(),
                            "category_id": classmap[_pred_class]["actual_category"],
                            "bbox": _pred_box,
                            "score": _score,
                        }
                    )

        with open("eval/results/results.json", "w") as f:
            json.dump(results, f)

    cocoGT = test_dataloader.dataset.coco
    cocoDT = cocoGT.loadRes("eval/results/results.json")
    coco_eval = COCOeval(cocoGT, cocoDT, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
