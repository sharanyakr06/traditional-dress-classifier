# Traditional Dress Classifier

A website that predicts whether an uploaded traditional-dress image is associated with Kerala, Punjab, or Rajasthan. The result shows just one state.

The website uses the trained model `resnet50_augmented_finetuned_best.keras`. It accepts JPG, JPEG, and PNG images up to 4 MB.

## Run locally

From PowerShell in this project folder:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Keep the terminal open while using the site.


### Why the photos are not pushed

The `kerala/`, `punjab/`, and `rajasthan/` folders contain the 485 example images used to prepare and train the classifier. The live website does not read those folders: it accepts a visitor's upload and makes a prediction with the already-trained model. Keeping the photos out makes this deployment repository smaller and avoids publishing the dataset. The photos remain in your local project.

The image folders are ignored by Git and stay local. Do not publish them unless you have the right to redistribute the photos. The CSV files, which describe the dataset splits, can be committed, but training requires the corresponding images to be present locally.

### What the `.keras` files are

`.keras` files contain saved neural-network models or training checkpoints; they are not source-code scripts. This project has 15 model files totaling about 1.6 GiB. Most are alternate architectures, final/best copies, or intermediate training stages made during experiments. **The website uses only `resnet50_augmented_finetuned_best.keras`**.

The other checkpoints are included for training and evaluation. All `.keras` files use Git LFS because several are larger than GitHub's regular file limit. Storing and downloading every checkpoint uses substantial Git LFS storage and bandwidth; they are not required to run the website.

## Model selection and accuracy

The website's selected model is **`resnet50_augmented_finetuned_best.keras`**. It is an ImageNet-pretrained ResNet50 backbone with a three-class head (global average pooling, a 128-unit ReLU layer, dropout, and a three-way softmax output). It expects ResNet50-preprocessed 224 x 224 RGB images; the Flask app applies that preprocessing before prediction. Training uses image augmentation, first trains the head with the backbone frozen, then fine-tunes the last 30 backbone layers while keeping batch-normalization layers frozen. The best fine-tuning checkpoint is selected by validation accuracy. The training script now reloads that best checkpoint before test evaluation and saves the same weights as `resnet50_augmented_finetuned_final.keras`.

All 15 saved checkpoint files were compared on the same 73-image `test.csv` using each architecture's matching input size and preprocessing. The highest score was **89.04% (65/73)** for `resnet50_augmented_finetuned_best.keras`, with macro F1 **0.8802**. The previously deployed `resnet50_improved_best.keras` scored **86.30% (63/73)**. Best observed results for the other model families were ResNet50 at 86.30%, MobileNetV2 at 76.71%, InceptionV3 at 76.71%, and VGG16 at 78.08%. One test image changes the score by about 1.37 percentage points.

On the selected model's test predictions, it classified Kerala **26/26**, Punjab **15/23**, and Rajasthan **24/24** correctly. Punjab recall is **65.22%**, so the overall accuracy should not be read as equally strong performance for every state.

The repeatable evaluation command is `python evaluate_resnet50_augmented.py`; it writes `resnet50_augmented_results.json` and `resnet50_augmented_confusion_matrix.csv`. Because the available test set is small and was used to compare these saved checkpoints, 89.04% is the best observed result for this project, not an unbiased guarantee of performance on new images. A separate unseen test set is needed for an independent final accuracy estimate.

## Project files

- `app.py` — Flask page and image-prediction endpoint.
- `templates/index.html` — website HTML.
- `public/` — website CSS and JavaScript served as static files.
- `train_*.py`, `evaluate_*.py`, and `predict*.py` — scripts for training, evaluating, or testing models; not used by the deployed website.
- `*_results.json` and `*_confusion_matrix.csv` — saved evaluation outputs.
