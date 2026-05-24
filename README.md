# Adversarial Examples Using Style Transfer

This project implements the coding challenge **"Adversarial Examples Using Style Transfer"** in PyTorch.

Given 10 content images and one or more style images, the goal is to generate adversarial styled images that:

1. Are classified by a fixed ImageNet-pretrained VGG19 model as the target class `movie theater, cinema`.
2. Preserve the main structure of the original content image.
3. Contain visual texture from the selected style image.

In this implementation, the VGG19 model is never trained. Instead, the image tensor `x_adv` itself is optimized directly with Adam.

## Method

The method uses a fixed torchvision VGG19 pretrained on ImageNet.

Two VGG19 wrappers are used:

- `VGG19Classifier`: returns ImageNet logits for adversarial classification.
- `VGG19FeatureExtractor`: returns intermediate VGG features for style and content losses.

All VGG19 parameters are frozen:

```python
param.requires_grad = False
```

The optimizer only receives the adversarial image variable:

```python
optimizer = torch.optim.Adam([x_adv], lr=args.lr)
```

The optimized image is kept in pixel range `[0, 1]`. Before being passed into VGG19, it is normalized with ImageNet mean and standard deviation.

The total objective is:

```text
total_loss =
    lambda_adv    * adversarial_loss
  + style_weight  * style_loss
  + content_weight * content_loss
  + tv_weight     * tv_loss
```

Loss terms:

- `adversarial_loss`: cross entropy loss toward the target ImageNet class `cinema`.
- `style_loss`: MSE between Gram matrices of VGG features from the adversarial image and the style image.
- `content_loss`: MSE between VGG content features from the adversarial image and the original content image.
- `tv_loss`: total variation loss for smoother images.

The default target class is automatically resolved from ImageNet labels by searching for `cinema` or `movie theater`. For torchvision VGG19, this is class index `498`.

## Project Structure

```text
adversarial_style_transfer_project/
+-- data/
|   +-- content/
|   +-- style/
|   +-- adv_style_dataset.py
|   +-- data_entry.py
+-- model/
|   +-- vgg19_wrapper.py
|   +-- model_entry.py
+-- utils/
|   +-- image_utils.py
|   +-- viz.py
|   +-- logger.py
|   +-- zip_utils.py
|   +-- imagenet_labels.py
+-- loss.py
+-- metrics.py
+-- options.py
+-- train.py
+-- eval.py
+-- scripts/
|   +-- run_smoke.sh
|   +-- run_attack.sh
|   +-- run_eval.sh
+-- requirements.txt
+-- README.md
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run requires the ImageNet-pretrained VGG19 weights. If the weights are not already cached, torchvision will try to download them automatically.

## Data Preparation

Place content images in:

```text
data/content/
```

Place style images in:

```text
data/style/
```

The dataset uses the first `--num_images` content images. Style images are assigned by:

```text
style_image = style_images[index % len(style_images)]
```

For the current setup, there are 10 content images and one selected fire-texture style image.

## Smoke Test

Run a quick test on one image for 50 optimization steps:

```bash
bash scripts/run_smoke.sh
```

Equivalent command:

```bash
python train.py --content_dir data/content --style_dir data/style --out_dir outputs --num_images 1 --steps 50 --target_name cinema --make_zip
```

## Full Attack

Run the attack on 10 images:

```bash
bash scripts/run_attack.sh
```

Equivalent command:

```bash
python train.py --content_dir data/content --style_dir data/style --out_dir outputs --num_images 10 --steps 1000 --target_name cinema --lambda_adv 100 --style_weight 200000 --content_weight 10 --tv_weight 0.0001 --lr 0.03 --make_zip
```

## Recommended Conservative Run

If the generated images change too much visually, use a lower style weight, higher content weight, lower learning rate, and early stopping:

```bash
python train.py --content_dir data/content --style_dir data/style --out_dir outputs --run_name fire_conservative_v1 --num_images 10 --steps 1000 --target_name cinema --lambda_adv 100 --style_weight 50000 --content_weight 50 --tv_weight 0.0003 --lr 0.01 --early_stop --make_zip
```

## Evaluation

Evaluate the latest run:

```bash
bash scripts/run_eval.sh
```

Equivalent command:

```bash
python eval.py --out_root outputs --target_name cinema
```

Evaluate a specific run:

```bash
python eval.py --run_dir outputs/runs/<run_name> --target_name cinema
```

## Output Organization

Each training run is saved in a separate directory:

```text
outputs/
+-- latest_run.txt
+-- runs/
    +-- <run_name>/
        +-- adv_images/
        +-- figures/
        +-- logs/
        +-- intermediate/
        +-- args.json
        +-- results.csv
        +-- eval/
        |   +-- eval_results.csv
        |   +-- result.txt
        +-- submission.zip
```

Important files:

- `adv_images/`: generated adversarial style-transfer images.
- `figures/`: side-by-side visualizations of content, style, and adversarial images.
- `logs/`: per-image optimization logs.
- `results.csv`: attack results from the training run.
- `eval/eval_results.csv`: evaluation results for generated images.
- `eval/result.txt`: summary with success count, success rate, average target probability, and per-image top-1 predictions.
- `submission.zip`: packaged outputs and source code, created when `--make_zip` is used.
- `outputs/latest_run.txt`: path to the most recent run.

If a manually specified `--run_name` already exists, the code automatically appends a suffix such as `_001` to avoid overwriting old results.

## Packaging

Add `--make_zip` when running `train.py`:

```bash
python train.py --content_dir data/content --style_dir data/style --out_dir outputs --num_images 10 --steps 1000 --target_name cinema --make_zip
```

This creates:

```text
outputs/runs/<run_name>/submission.zip
```

## Notes

This README does not assume a fixed attack success rate. Actual results should be reported from `results.csv` and `eval/result.txt`.
