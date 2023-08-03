import json
import os
import shutil
import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

''' This file defines loading OWLVIT with PTL and training in that ecosystem.'''

import torch
import pytorch_lightning as pl
from PIL import Image
from transformers import AutoProcessor, OwlViTForObjectDetection
from src.losses import PushPullLoss
from src.models import PostProcess, OwlViT
from src.train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics,
)
from src.util import BoxUtil, GeneralLossAccumulator


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


class OwlVITModule(pl.LightningModule):
    def __init__(self, training_cfg,scales=None,labelmap=None):
        super().__init__()
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
     



        self.postprocess = PostProcess(
                                        confidence_threshold=training_cfg["confidence_threshold"],
                                        iou_threshold=training_cfg["iou_threshold"],
                                    )

    
        self.training_cfg = training_cfg
        self.metric=MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        self.epoch_train_losses = GeneralLossAccumulator()
        self.train_loss_accumulator, self.val_loss_accumulator = [GeneralLossAccumulator()] * 2
    def forward(self, x):
        return self.model(x)
    def on_train_epoch_start(self):
        self._processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        
        labels=self.trainer.datamodule.labels
        print(labels)
        scales=self.trainer.datamodule.scales if self.trainer.datamodule.scales else None
        
        inputs = self._processor(
            text=[list(labels)],  
            images=Image.new("RGB", (224, 224)),
            return_tensors="pt",
        )
        
        with torch.no_grad():
            self.queries = self.model(**inputs).text_embeds

        
        self.model = OwlViT(pretrained_model=self.model, query_bank=self.queries)
    
        self.criterion = PushPullLoss(
                                        len(labels),
                                        scales=scales,
                                        )
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        
        image,labels,boxes,metadata=batch
        all_pred_boxes, _, pred_sims, _ = self(image)
        losses = self.criterion(pred_sims, labels, all_pred_boxes, boxes)
        loss = (
            losses["loss_ce"]
            + losses["loss_bg"]
            + losses["loss_bbox"]
            + losses["loss_giou"]
        )
        self.log("train_loss", loss)
        self.log("train_loss_ce", losses["loss_ce"])
        self.log("train_loss_bg", losses["loss_bg"])
        self.log("train_loss_bbox", losses["loss_bbox"])
        self.log("train_loss_giou", losses["loss_giou"])

        return loss
    def on_validation_epoch_start(self):
        self.labelmap = self.trainer.datamodule.val.labelmap
    def validation_step(self, batch, batch_idx):
        image,labels,boxes,metadata=batch
        # Get predictions and save output
        pred_boxes, _, pred_class_sims, _ = self(image)
        losses = self.criterion(pred_class_sims, labels, pred_boxes, boxes)
        self.val_loss_accumulator.update(losses)

        pred_boxes, pred_classes, scores = self.postprocess(
            pred_boxes, pred_class_sims
        )

        # Use only the top 200 boxes to stay consistent with benchmarking
        top = torch.topk(scores, min(200, scores.size(-1)))
        scores = top.values
        inds = top.indices.squeeze(0)

        update_metrics(
            self.metric,
            metadata,
            pred_boxes[:, inds],
            pred_classes[:, inds],
            scores,
            boxes,
            labels,
        )

        if self.training_cfg["save_eval_images"]:
            pred_classes_with_names = labels_to_classnames(
                pred_classes, self.labelmap
            )
            pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
            image_with_boxes = BoxUtil.draw_box_on_image(
                metadata["impath"].pop(),
                pred_boxes,
                pred_classes_with_names,
            )

            write_png(image_with_boxes, f"debug/{self.current_epoch}/{batch_idx}.jpg")
        return losses
    
    def validation_epoch_end(self, outputs):
        classMAPs = {v: [] for v in list(self.labelmap.values())}
        #epoch_val_losses = val_loss_accumulator.get_values()

        val_metrics = self.metric.compute()
        for i, p in enumerate(val_metrics["map_per_class"].tolist()):
            label = self.labelmap[str(i)]
            classMAPs[label].append(p)

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        self.metric.reset()


    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(
            [p for n,p in self.model.named_parameters() if any( [
            "box" in n,
            "post_layernorm" in n,
            "class_predictor" in n,
            "queries" in n,
        ]) ],
            lr=float(self.training_cfg["learning_rate"]),
            weight_decay=self.training_cfg["weight_decay"],
            amsgrad=True,
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    
    from src.dataset import COCODataModule
    training_cfg = get_training_config()
    datamodule=COCODataModule(dir="/data",batch_size=1)
    # train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    module = OwlVITModule(training_cfg)


    trainer = pl.Trainer(
                         precision=16,
                         max_epochs=2,#args['epochs'], 
                         num_sanity_val_steps=0,
                         gradient_clip_val=0.25,
                         accumulate_grad_batches=4,
                         #callbacks=[ModelCheckpoint(dirpath=args['output_dir'],save_top_k=1,monitor='val_loss',mode='min')],
                         accelerator='auto',
                         fast_dev_run=True,  
                         devices="auto",
                            )
    trainer.fit(module,datamodule)
    trainer.test(module,datamodule)

