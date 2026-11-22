# References: ignored pixels in the labels and in the training loss

Why the labels leave some pixels undecided (`255`), why the U-Net is trained without them and still predicts them,
and the literature that supports each step. Collected for the paper on 22 November 2026.

## What the project does

- **Labels.** A pixel is labelled `255` (ignore) when it is untrusted (twilight, a missing channel) or when its
  brightness-temperature difference lies within ±0.5 K of the 4.0 K threshold, too close to call. Pixels under a cold
  or cirrus top are `2` (obscured). See [`DATASET_CARD.md`](../DATASET_CARD.md).
- **Training.** Pixels labelled `255` or `2` add nothing to the U-Net's loss. The network is told neither "fog" nor
  "no fog" there.
- **Prediction.** The U-Net predicts every trusted, non-obscured pixel, including those that were `255` in its
  training labels; its labels have no ignore band.
- **Evaluation.** Those pixels carry no label, so the predictions are scored only against independent station
  observations on held-out nights and stations ([`better_labels.md`](better_labels.md)). The full count scores a
  `255` under a fog report as a miss, so a rule cannot gain by leaving hard pixels undecided.

## Claims and sources

### 1. "255 = ignore" is the standard convention in semantic segmentation

| Source | What it says | Checked |
|---|---|---|
| Cityscapes label definitions, Cordts et al., [`cityscapesScripts/helpers/labels.py`](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) | Void classes ("unlabeled", "ego vehicle", "ground", …) have `trainId` 255 and `ignoreInEval` True. The file says: *"Note that you might want to ignore labels with ID 255 during training."* | Primary source read (lines 57 and 64–78) |
| PASCAL VOC 2012, Everingham et al., devkit documentation | Object borders and difficult regions are labelled "void", index 255, and not evaluated | **To check:** confirmed only by secondary sources ([CascadePSP, arXiv 2005.02551](https://arxiv.org/pdf/2005.02551), [dlib VOC reader](https://dlib.net/pascal_voc_2012.h.html)). Quote the segmentation section of the [devkit PDF](https://datasets.cms.waikato.ac.nz/ufdl/data/pascalvoc2012/devkit_doc.pdf) before citing |
| PyTorch, [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/2.6/generated/torch.nn.CrossEntropyLoss.html) | `ignore_index`: *"Specifies a target value that is ignored and does not contribute to the input gradient."* The mean is taken over non-ignored targets only | Primary source read |

### 2. Training on the labelled pixels only and predicting all of them is an established method

- **Bearman, A., Russakovsky, O., Ferrari, V., Fei-Fei, L. (2016).** What's the point: Semantic segmentation with
  point supervision. *ECCV 2016*, LNCS, pp. 549–565. doi:10.1007/978-3-319-46478-7_34.
  [arXiv:1506.02106](https://arxiv.org/abs/1506.02106). A segmentation CNN trained with one labelled point per object;
  the loss is computed at the points, the network predicts every pixel. +12.9 % mIoU over image-level supervision on
  VOC 2012.
- **Tang, M., Perazzi, F., Djelouah, A., Ben Ayed, I., Schroers, C., Boykov, Y. (2018).** On regularized losses for
  weakly-supervised CNN segmentation. *ECCV 2018*. [arXiv:1803.09569](https://arxiv.org/abs/1803.09569). Names the
  loss over labelled pixels only "partial cross-entropy". With scribbles alone: 55.8 % mIoU on VOC 2012 val, against
  63.0 % with full masks ([rloss repository](https://github.com/meng-tang/rloss)).
- **Remote sensing.** [Sparse point annotations for remote sensing image segmentation, *Scientific Reports* (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12301467/);
  [Weakly supervised deep learning for segmentation of remote sensing imagery, *Remote Sensing* 12(2):207 (2020)](https://www.mdpi.com/2072-4292/12/2/207),
  which masks unlabelled pixels "similar to how pixels with the 'void' class label are masked out when training on
  the Pascal VOC segmentation dataset".
- Further point and partial losses in [`label_improvement_research.md`](label_improvement_research.md) (a frost
  segmentation model trained at observatory pixels, partial cross-entropy at IGARSS 2024, LESS); not re-checked here.

### 3. The uncertainty band is a reject option

- **Chow, C. K. (1970).** On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*
  16(1), 41–46. doi:10.1109/TIT.1970.1054406. The classic analysis of a classifier that abstains when unsure, and of
  the trade-off between its error rate and how often it abstains. The ±0.5 K band applies such a rejection to the
  labels.

### 4. Networks can be trained on imperfect labels, with a limit

- **Rolnick, D., Veit, A., Belongie, S., Shavit, N. (2018).** Deep learning is robust to massive label noise.
  [arXiv:1705.10694](https://arxiv.org/abs/1705.10694). High accuracy with many randomly wrong labels; more noise
  needs more data.
- **Song, H., Kim, M., Park, D., Shin, Y., Lee, J.-G. (2023).** Learning from noisy labels with deep neural networks:
  A survey. *IEEE Transactions on Neural Networks and Learning Systems* 34(11), 8135–8153 (online 2022).
  doi:10.1109/TNNLS.2022.3152527. [arXiv:2007.08199](https://arxiv.org/abs/2007.08199).

**Limit:** these results concern random noise. The errors of the rule labels are systematic (false fog at steep
viewing angles in the north and east, missed broken decks), and a network learns systematic errors too. The U-Net
evaluation of the later phases agrees: outside the teacher's ignore band, the U-Net and its training labels score
alike. Do not cite these papers as evidence that the network corrects the labels.

## What a reviewer is likely to ask

1. **"The ignored pixels are not missing at random; they are the hard cases."** True. The papers above mostly ignore
   pixels for other reasons (unannotated, object borders). The answer is the evaluation: predictions in those pixels
   are scored only against station observations on held-out nights and stations, never against the labels.
2. **"Is there precedent in fog or cloud detection?"** None found for an ignore band in weak fog labels. Closest:
   cloud-removal work that keeps the uncertain zone as its own class (opaque, transparent and no cloud;
   [*Remote Sensing* 15(4):904 (2023)](https://www.mdpi.com/2072-4292/15/4/904)), and a patent that leaves an
   unknown zone around targets out of the loss (not peer-reviewed). Present the method as a standard segmentation
   practice applied to weak physical labels, not as established practice in fog detection.

## Draft text for the methods section

> Pixels whose rule label is uncertain (brightness-temperature difference within ±0.5 K of the threshold), untrusted
> (twilight or missing channels) or obscured were excluded from the training loss, following the "void" (255)
> convention of PASCAL VOC and Cityscapes and the partial cross-entropy used in weakly supervised segmentation
> (Bearman et al., 2016; Tang et al., 2018). The uncertainty band acts as a reject option on the labels (Chow, 1970).
> Because these pixels carry no label, the network's predictions there were evaluated only against independent
> station observations on held-out nights and stations.

## Before submitting

- [ ] Quote the PASCAL VOC devkit's own wording on "void" (segmentation section).
- [ ] Add the fog and low-stratus references the results are compared with: Egli et al. (2017), Cermak & Bendix
  (2008), Jahani et al. (2025); full entries in [`station_validation.md`](station_validation.md).
- [ ] Re-check the references in [`label_improvement_research.md`](label_improvement_research.md) before citing them.
