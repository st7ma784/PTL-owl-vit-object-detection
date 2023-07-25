import json
import os
import shutil

import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

from src.losses import PushPullLoss
from src.dataset import get_dataloaders
from src.models import PostProcess, load_model
from src.train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics,
)
from src.util import BoxUtil, GeneralLossAccumulator, ProgressFormatter


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


def init_weights(m):
    # frozen = []
    # for parameter in m.parameters():
    #     frozen.append(parameter.requires_grad)

    # print(frozen)
    # print()
    if isinstance(m, torch.nn.Linear):
        # print("WEIGHTS RE INITIALIZED")
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        print(dir(m))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    epoch_train_losses = GeneralLossAccumulator()
    train_loss_accumulator, val_loss_accumulator = [GeneralLossAccumulator()] * 2
    progress_summary = ProgressFormatter()

    if os.path.exists("debug"):
        shutil.rmtree("debug")

    training_cfg = get_training_config()
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    model, trainable = load_model(labelmap, device)

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    criterion = PushPullLoss(
        len(labelmap),
        scales=torch.tensor(scales).to(device)
        if training_cfg["use_class_weight"]
        else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    classMAPs = {v: [] for v in list(labelmap.values())}
    model.train()
    for epoch in range(training_cfg["n_epochs"]):
        print(scheduler.get_lr())
        # scheduler.step()
        # continue
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
            # train_dataloader
        ):
            optimizer.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Predict
            all_pred_boxes, _, pred_sims, _ = model(image)
            losses = criterion(pred_sims, labels, all_pred_boxes, boxes)
            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                + losses["loss_bbox"]
                + losses["loss_giou"]
            )
            loss.backward()
            # print(loss.sum().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_loss_accumulator.update(losses)

        epoch_train_losses = train_loss_accumulator.get_values()
        scheduler.step()
        # Eval loop
        model.eval()
        with torch.no_grad():
            for i, (image, labels, boxes, metadata) in enumerate(
                tqdm(test_dataloader, ncols=60)
            ):
                # Prep inputs
                image = image.to(device)
                labels = labels.to(device)
                boxes = coco_to_model_input(boxes, metadata).to(device)

                # Get predictions and save output
                pred_boxes, _, pred_class_sims, _ = model(image)
                losses = criterion(pred_class_sims, labels, pred_boxes, boxes)
                val_loss_accumulator.update(losses)

                pred_boxes, pred_classes, scores = postprocess(
                    pred_boxes, pred_class_sims
                )

                # Use only the top 200 boxes to stay consistent with benchmarking
                top = torch.topk(scores, min(200, scores.size(-1)))
                scores = top.values
                inds = top.indices.squeeze(0)

                update_metrics(
                    metric,
                    metadata,
                    pred_boxes[:, inds],
                    pred_classes[:, inds],
                    scores,
                    boxes,
                    labels,
                )

                if training_cfg["save_eval_images"]:
                    pred_classes_with_names = labels_to_classnames(
                        pred_classes, labelmap
                    )
                    pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                    image_with_boxes = BoxUtil.draw_box_on_image(
                        metadata["impath"].pop(),
                        pred_boxes,
                        pred_classes_with_names,
                    )

                    write_png(image_with_boxes, f"debug/{epoch}/{i}.jpg")

        epoch_val_losses = val_loss_accumulator.get_values()

        print("Computing metrics...")
        val_metrics = metric.compute()
        for i, p in enumerate(val_metrics["map_per_class"].tolist()):
            label = labelmap[str(i)]
            classMAPs[label].append(p)

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        metric.reset()
        progress_summary.update(
            epoch, epoch_train_losses, epoch_val_losses, val_metrics
        )

        progress_summary.print()
