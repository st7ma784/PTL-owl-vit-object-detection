**NOTICE: I'll be updating this repo periodically until I'm happy with it, at which point I'll make a release. Please watch and star to stay up to date.**
# TODO
- [x] **Introduce a learnable query bank.** The original Owl-VIT model is multi-modal, taking an image and a prompt as input. Now, the model computes a set of initial queries based on the labels provided which become a parameter of the model. The queries are injected during each forward pass, optimal queries for each class are learned through training. See `experiments/check_text_embeddings_as_priors.ipynb` for more details.
- [x] **Get rid of pycocotools.** It is annoying.
- [x] **Half precision training.** For speed.
- [ ] **Benchmarks.** Do some benchmarking writeups.
- [x ] **Improve model.** Currently, while the model is functional, it isn't optimal. There seem to be some issues due to the fact that a ton of boxes end up clustered around the object of interest (see `experiments/check_zero_shot_results.ipynb`) that the model is having trouble learning through but this requires more investigation.
- [ ] **Batching.** Everything right now only works with batchsize of 1. Eventually I'll generalize to any batchsize
- [ ] **More ViT Backbones.** There are others available that may improve results

# Preliminary Results
On a small sample of COCO, I am able to achieve a mAP of 0.415 in just 5 epochs (~10 minutes of actual train time, ~5 minute val/metric computation). Compare this to DETR's mAP of 0.42 (per their repo, although the results aren't directly comparable since they are not subsetting COCO).
```
100%|███████████████████| 2485/2485 [02:14<00:00, 18.42it/s]
100%|███████████████████████| 99/99 [00:04<00:00, 22.73it/s]
Computing metrics...

  epoch  class loss (T/V)    bg loss (T/V)      map    map@0.5  map (L/M/S)     mar (L/M/S)     time elapsed
-------  ------------------  ---------------  -----  ---------  --------------  --------------  --------------
      0  0.4948/0.4908       0.3713/0.3667    0.405      0.582  0.59/0.37/0.23  0.72/0.5/0.29   0:03:04
      1  0.4691/0.4674       0.3064/0.3043    0.41       0.604  0.58/0.39/0.23  0.68/0.51/0.3   0:06:11
      2  0.4554/0.4542       0.2749/0.2737    0.415      0.61   0.57/0.4/0.21   0.68/0.51/0.28  0:09:13
      3  0.445/0.4443        0.2561/0.2552    0.408      0.604  0.55/0.39/0.22  0.67/0.5/0.3    0:12:16
      4  0.4368/0.4362       0.2436/0.243     0.415      0.617  0.57/0.4/0.22   0.69/0.51/0.29  0:15:14
```
# Motivation
Models like CLIP and Owl-VIT are interesting to me because of the massive amount of data that they've been trained on. CLIP's usefulness in the scope of computer vision is limited since it can only handle classification. The ability for Owl-VIT to localize unseen objects extremely well in image-guided one/few shot tasks is impressive, but it still relies on the presence of a query. The idea in this repo is to repurpose Owl-VIT for a traditional object detection task, since its massive pre-training should (hopefully) allow the model to produce good results with much less data than you'd typically need.

# Installation
I recommend using miniconda to handle environments, the setup scipt I've written depends on conda. Setup the environemnt as follows:

```
conda create -n owl python=3.10
conda activate owl
make install
```

After this, your environment should be good to go. Next retrieve the data. Download the coco 2014 train subset from here (images and annotations respectively):
```
http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Unzip them to wherever you want and modify the paths in `config.yaml` to point to the correct locations. **NOTE** I'm only using the train subset to decrease the amount of data I need. Since the goal is to produce a good few-shot model, it's no problem to use a small subset of the train and test on another small subset.

Once you have everything unzipped, run:

```
python scripts/make_coco_subset.py
```

Which will randomly sample the number of instances specified in the yaml until the user accepts the distribution label distribution. It will probably be extremely out of balance but that's okay this is handled during training with scaling.

Now you're ready to train with

```
python main.py
```

During training, for class-level loss plots, run
```
tensorboard --logdir=logs
```
![image](assets/TensorBoard.png "Tensorboard Screenshot")
