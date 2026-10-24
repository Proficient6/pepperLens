# PepperLens — bell pepper leaf classification through DSP

The final project consists of the self-contained Colab training notebook, the website, and your actual six-class / seven-classifier trained model. You do not need to retrain before using the website. The website can run locally or be deployed to Render using the included configuration.

PepperLens uses digital image processing to turn bell pepper leaf photographs into colour, texture and shape descriptors, then compares seven classical classifiers. Its browser interface supports image upload, processing-stage inspection and the complete saved training report. The saved main model is Random Forest, selected on validation balanced accuracy.

## Quick start and prerequisites

- **Windows:** Python 3.12 and a browser. The included launcher handles first-time dependency installation.
- **GitHub setup:** Git and a GitHub account. The trained models and reports are already included in `website/trained_run/`.
- **Training:** a hosted Google Colab CPU runtime. The notebook fetches training data through its embedded API.

Open PowerShell or Windows Terminal and enter the folder containing this README and `Start-PepperLens.cmd`:

```powershell
# Replace this example with the actual location of your project folder.
Set-Location -LiteralPath "C:\path\to\bell paper disease"
Get-Location
Get-ChildItem -Name
```

You should see `Start-PepperLens.cmd`, `PepperLens_FINAL_DSP_Colab.ipynb`, `pepper_model.py` and `website`. All commands below assume this directory. To open PowerShell in the right location without typing its path, open the project folder in File Explorer, click the address bar, type `powershell` and press Enter.

Start the app from that terminal:

```powershell
.\Start-PepperLens.cmd
```

Then use http://127.0.0.1:7860. Keep the server terminal open while making predictions.

## Important files

```text
bell paper disease/
├── README.md                           This single guide
├── .python-version                     Python 3.12 for Render
├── render.yaml                         Render deployment configuration
├── Start-PepperLens.cmd                 Start the website on Windows
├── PepperLens_FINAL_DSP_Colab.ipynb      Complete training notebook, with dataset API
├── pepper_model.py                     DSP, training and inference implementation
├── dsp_walkthrough.py                  Processing / Fourier / feature plots
└── website/
    ├── server.py                       Website backend and model loading
    ├── wsgi.py                         Flask entry point for public hosting
    ├── static/                         HTML, CSS, JavaScript and icon
    ├── trained_run/                    YOUR TRAINED MODELS + original reports/figures
    ├── requirements-tested.txt         Matching dependencies
    ├── requirements-production.txt     Matching dependencies + production servers
    ├── import_run.py                   Import a later Colab output ZIP
    ├── test_website.py                 Optional inference smoke test
    └── .venv/                          Installed website environment
```

**Keep both the notebook and saved model.** The notebook contains the executable training procedure. `website/trained_run/pepper_models.joblib` contains the fitted estimators and scalers from your Colab run. Keep the entire `trained_run` directory so the website can show all supporting reports and figures.

The hidden `.git` folder and `.gitignore` preserve version-control metadata. Python/tests may recreate `__pycache__` or `website/runtime`; those are generated files.

## Run the website

1. Double-click **Start-PepperLens.cmd**.
2. Open **http://127.0.0.1:7860** if your browser does not open automatically.
3. Keep its terminal window open. Drop/browse for one bell pepper leaf and click **Analyze leaf**.
4. Ctrl+C in the terminal stops the server.

The website environment is installed on this computer. On a fresh computer, install Python 3.12; the launcher creates the environment and installs its packages once. The website requires no dataset download or retraining.

Manual startup from this project folder:

```powershell
.\website\.venv\Scripts\python.exe -X utf8 website\server.py
```

If the environment is missing, first run:

```powershell
py -3.12 -m venv website\.venv
.\website\.venv\Scripts\python.exe -m pip install -r website\requirements-tested.txt
```

If another app occupies port 7860, add `--port 7861` and use http://127.0.0.1:7861. This is a local website. Uploaded photographs are processed in memory and are not saved by the app.

The website includes the selected main prediction, all seven classifier results, available model scores, fresh RGB/BGR/HSV/filter/mask/segmentation/edge/Fourier images, feature tables and downloadable prediction JSON. The training dashboard shows your original Colab figures, model comparisons, selectable confusion matrices, per-class metrics, data splits and cleaning audits. The DSP view explains the processing pipeline.

## Run or study the notebook in Colab

1. Open https://colab.research.google.com/ and upload **PepperLens_FINAL_DSP_Colab.ipynb**.
2. Connect a hosted CPU runtime; no GPU is needed.
3. Choose **Runtime → Run all**. The notebook installs dependencies, reconstructs its embedded helper code and fetches the dataset into the cloud runtime. No separate Python files or manual training-image uploads are required.
4. Inspect cleaned counts, group partitions, validation/test scores and the DSP plots.
5. Save **PepperLens_FINAL_results.zip** before the temporary runtime expires.

Default source: [Pepper Bell Leaf Disease v1](https://data.mendeley.com/datasets/w9drvtxg9g/1). Its archive API is embedded in the notebook. Compatible data/features are reused within a cloud session; a runtime reset normally removes `/content` data. External source availability is outside the notebook's control.

To use a later Colab result, stop the website and import its output ZIP:

```powershell
.\website\.venv\Scripts\python.exe -X utf8 website\import_run.py "C:\path\to\PepperLens_FINAL_results.zip"
```

Restart the website afterward. Preserve the old output ZIP separately if comparing experiments. The importer checks feature-code compatibility. If a future Colab run changes scikit-learn versions, update the website environment to match that run; the server rejects a mismatch.

## Understand the DSP first

Study notebook sections **A → B → C → D**. Then trace these functions in `pepper_model.py`: `read_rgb`, `preprocess`, `extract_features`, `make_manifest`, `assign_splits`, `train`, and `predict_all`.

| Operation | DSP / image-processing concept | Role |
|---|---|---|
| Decode/orient, resize to 256 × 256 | Sampled 2-D signal / spatial resampling | Consistent orientation and pixel grid |
| BGR ↔ RGB | Channel-order representation | Prevent colour interpretation errors; Pillow supplies RGB directly |
| 3 × 3 Gaussian smoothing, σ = 0.7 | 2-D FIR convolution / low-pass filtering | Reduce fine spatial intensity changes |
| RGB → HSV | Nonlinear colour-coordinate transform | Separate hue, saturation and value for thresholds |
| Threshold, OR, closing, contour filling | Segmentation / mathematical morphology | Select a candidate leaf region, retaining internal lesions |
| Masked 8 × 8 × 8 HSV histogram | Quantization / colour-distribution statistics | 512 colour features |
| Haralick descriptors | Gray-level co-occurrence / spatial statistics | 13 texture features |
| Log-transformed Hu moments | Normalized spatial moments | Seven shape features |

Every classifier receives **512 + 13 + 7 = 532 features**. Where needed, scaling is fitted inside training-only pipelines. The seven algorithms are Logistic Regression, LDA, KNN, Decision Tree, Random Forest, Gaussian Naive Bayes and SVM. Validation balanced accuracy selects the main model before test evaluation. Voting is a separate comparison.

The Gaussian kernel/frequency response, before/after spectra and Sobel gradients explain the signal processing. DFT/Sobel plots are diagnostics, not extra classifier features. The DCT-based perceptual hash is for duplicate grouping only. No CNN, wavelet classifier or bag-of-visual-words implementation is claimed.

Dataset cleaning and image preprocessing are separate. Cleaning audits unreadable files, excludes every copy of conflicting exact images, removes redundant same-label copies and keeps detected groups together across splits. Preprocessing smooths and segments each image before measurement. Source photographs are preserved.

## Your completed experiment

Imported from **PepperLens_FINAL_results (1).zip**; its source hash is recorded in `website/trained_run/import_provenance.json`.

| Measurement | Recorded result |
|---|---:|
| Readable source files | 9,283 |
| Conflicting-label copies excluded | 48 |
| Same-label duplicate copies removed | 1,029 |
| Retained unique images | 8,206 |
| Train / validation / test before feature filtering | 4,939 / 1,631 / 1,636 |
| Usable training rows | 4,938; one feature-extraction failure |
| Selected classifier | Random Forest |
| Validation balanced accuracy | 99.2463% |
| Test accuracy | **99.3888% — 1,626 / 1,636 correct** |
| Test balanced accuracy | 99.4364% |
| Test macro F1 | 99.0889% |

Classes: healthy, bacterial spot, Cercospora leaf spot, leaf curl, nutrient deficiency and powdery mildew. Nutrient deficiency is abiotic; leaf curl is the dataset's symptom label. Crop identity and unseen conditions are not verified.

**Confidence is not per-photo accuracy.** An uploaded image's percentage is an uncalibrated class score. Test accuracy is measured across labelled test images. SVM supplies no probability percentage. Mask coverage is not disease severity. Field accuracy remains unmeasured, and hashes cannot establish complete independence of all original/augmented images.

Training used Python 3.13.16, sklearn 1.6.1, NumPy 2.1.3, OpenCV 5.0.0 and mahotas 1.4.19. The website uses Python 3.12 with matching sklearn/NumPy and the tested dependency list. Saved-model inference, browser upload, DSP figures and report downloads were verified; the website independently reproduces the correct-test-image count from `test_predictions.csv`.

Optional smoke test, with no dataset needed:

```powershell
.\website\.venv\Scripts\python.exe -X utf8 website\test_website.py
```

The default generates a leaf-like signal to test integration, not disease accuracy. Add `--image "C:\path\to\leaf.jpg"` for your own photograph, or `--base-url http://127.0.0.1:7860` to check a running website. For your coursework, understand the features and later collect an independently labelled field test set before claiming real-world accuracy.

## Copy or submit the project

Include this notebook, README, launcher, the two shared Python modules and the entire `website` folder **except `.venv`, `__pycache__` and `runtime`**. Preserve `website/trained_run`, especially `pepper_models.joblib`. Dependencies can be installed again; cloud training data can be fetched from the notebook's API.

Only load trusted joblib/pickle artifacts. Your original output ZIP in Downloads and original notebook on Desktop are outside this cleaned folder and were left untouched.

## Troubleshoot uploads and new photographs

**“Failed to fetch” is a browser/server connection failure, not evidence that a leaf is absent from the training dataset.** The page can remain visible after its Python server stops. Start `Start-PepperLens.cmd`, keep its terminal open, and click **Check connection** in the website's offline message. The app checks connection status periodically and before prediction. An already-selected photograph is retained through reconnection; there is no need to retrain or add it to the dataset.

Other failures have different meanings:

| What happens | Meaning / action |
|---|---|
| Cannot reach server | Start the launcher and keep the backend running; a browser tab alone cannot perform prediction. |
| Server response times out | Check the terminal and retry after processing finishes; try a smaller image if needed. |
| Unreadable image | Export the photograph as JPG/PNG/WebP; changing the filename extension does not convert it. |
| Upload over 15 MB or 25 megapixels | Resize/export the photograph before upload. |
| Too little usable leaf colour | Try one close, well-lit leaf on a plain background; inspect the mask. |
| A class is predicted with a low score | The model ran, but the classification needs review. This is not a network error. |

New photographs are the intended use of the classifier. However, outdoor backgrounds, shadows, camera differences, multiple leaves, severe colour changes and untrained diseases can reduce performance or break heuristic segmentation. The classifier can assign a known class even to unfamiliar input. A technical fix does not establish field accuracy. Collect independently labelled photographs, inspect their masks, report per-class errors/recall, and use that evidence to decide whether segmentation, training diversity or score calibration needs improvement. Keep a fresh final test set after further model development.

## Put this project on GitHub

1. Sign in to GitHub and create an empty repository, for example `bell-pepper-dsp`. Do not prefill another README or `.gitignore` for this first push.
2. Open PowerShell in this project folder using the Quick start instructions. Check `git --version`. Git is already initialized in the original project folder; for an extracted copy without `.git`, run `git init` once.
3. Run the following commands, replacing `YOUR_USERNAME` with your GitHub account. Authenticate through Git/GitHub's normal prompt if requested. The repository name must match the empty repository you created.

```powershell
git status
git add .
git commit -m "Add PepperLens DSP project and trained model"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bell-pepper-dsp.git
git push -u origin main
```

Before adding a remote, inspect `git remote -v`. If `origin` already points to your intended repository, skip `git remote add`. Use `git add`, `git commit` and `git push` for subsequent changes.

If Git reports **Author identity unknown**, set the identity you want on your commits, then retry the commit. Use your real details or the GitHub-provided no-reply email shown in your account's email settings:

```powershell
git config user.name "Your Name"
git config user.email "YOUR_COMMIT_EMAIL"
```

The included `.gitignore` excludes local Python environments, generated caches/screenshots/logs, downloaded ZIPs, old dataset/model-cache folders and `.env` secrets. It deliberately **keeps** the final notebook, current `.joblib` model, original reports and training plots. Review `git status` after `git add .` before committing.

Include the final notebook, source, launcher, this guide and `website/trained_run`. The `.gitignore` excludes environments, caches, runtime files and `.env` credentials. Do not upload a training dataset. Retain the dataset citation above. The current joblib model is approximately 48 MB: use Git commands because GitHub's browser upload limit is 25 MiB; ordinary Git rejects individual files above 100 MiB. Consider a release artifact or LFS if future models grow, and ensure a host downloads the actual model rather than an LFS pointer.

Sources: [GitHub's existing-project push guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github), [GitHub file-size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Deploy PepperLens to Render, step by step

GitHub stores the project. **GitHub Pages alone cannot execute this Python/scikit-learn pipeline**; Pages serves static HTML/CSS/JavaScript. Keep the frontend and API on the same Python service for a straightforward first deployment. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

Your repository is [Proficient6/pepperLens](https://github.com/Proficient6/pepperLens). The new `website/wsgi.py` exposes a Flask app that reuses the saved classifiers, DSP pipeline and reports. Render runs it with Gunicorn. The Windows launcher continues to use the local server.

### 1. Open the project and install the hosting dependencies

Run in PowerShell on this computer:

```powershell
Set-Location -LiteralPath "C:\Users\Parth Kishan\OneDrive\ドキュメント\ChatGPT\bell paper disease"
.\website\.venv\Scripts\python.exe -m pip install -r website\requirements-production.txt
.\website\.venv\Scripts\python.exe -X utf8 website\test_website.py --production
```

Continue after the test prints `"status": "passed"` and `"production_checks": true`. If it fails, resolve the printed error before pushing. No images or training dataset are downloaded by this test. It uses a generated leaf-like signal to check integration, not disease accuracy. The same production test runs during Render's build, so a failing check prevents the new version from deploying.

Optional browser check of the production entry point on Windows:

```powershell
.\website\.venv\Scripts\python.exe -m waitress --listen=127.0.0.1:7861 --threads=4 website.wsgi:app
```

Open http://127.0.0.1:7861, upload one of your photographs and check the training and DSP tabs. Keep this terminal open; Ctrl+C stops this test server. This leaves port 7860 available for the usual launcher. Gunicorn is installed on Linux only; Waitress is used for this Windows check.

### 2. Push the deployment changes to your existing GitHub repository

From the project folder, check `git remote -v` still shows `https://github.com/Proficient6/pepperLens.git`, then run:

```powershell
git status
git add README.md .python-version render.yaml website/wsgi.py website/requirements-production.txt website/test_website.py website/static/app.js
git diff --cached --stat
git commit -m "Prepare PepperLens for Render deployment"
git push origin main
```

Confirm `render.yaml`, `.python-version` and `website/wsgi.py` appear on GitHub. Keep the repository root as the deployment root; setting it to `website` would hide the shared DSP modules. Your existing saved model and reports are already tracked. Do not upload `.venv` or a dataset, and do not run Colab training on Render.

### 3. Connect Render and deploy the Blueprint

1. Open the [Render Dashboard](https://dashboard.render.com/) and sign in.
2. Choose **New → Blueprint**.
3. Connect GitHub if prompted. Give the Render integration access to your `pepperLens` repository. If the repository is missing, check the integration's allowed repositories on GitHub.
4. Connect **Proficient6/pepperLens**. Use branch **main**, Blueprint name **PepperLens** and Blueprint path **render.yaml**.
5. Review the proposed service: **Python web service**, name **pepperlens**, plan **Free**. There is no database, disk or secret required for this app. Review the displayed plan before deploying; a paid upgrade is a separate decision.
6. Click **Deploy Blueprint**. Open the created service's **Events/Logs**. The build installs dependencies and runs the production smoke test, then starts Gunicorn and loads the saved model.
7. Wait for **Live**. Open the actual HTTPS `onrender.com` URL shown by Render; the exact hostname is assigned during deployment.

These steps follow [Render's Blueprint setup](https://render.com/docs/infrastructure-as-code) and [Flask deployment guide](https://render.com/docs/deploy-flask). Do not create a second Web Service if the Blueprint has already created one.

### 4. Verify the public app

1. Open `<your-render-url>/api/health`. Expect `application: "PepperLens"`, `status: "ready"`, `mode: "production"` and `selected_model: "Random Forest"`. Its `revision` identifies the deployed Git commit.
2. Open the homepage, upload a new bell pepper leaf and click **Analyze leaf**. Check the main result, all seven classifier results, 15 stage previews and three fresh DSP figures.
3. Check **Training report**: 1,626 / 1,636 correct, 99.3888% saved test accuracy, the original graphs and report download.
4. Try an unreadable image and a second simultaneous prediction. Expect a readable error rather than a crashed server. One image is analyzed at a time; a simultaneous request receives a retry message. Health and report requests remain available.
5. Check Render's logs and memory metrics during a large photograph and repeated predictions. A 48 MB model file does not imply 48 MB RAM. If the service is killed for memory use, resize the photograph and assess a larger instance before calling the deployment reliable.

You can also run the HTTP checks from your local project folder after replacing the example URL with the actual service URL:

```powershell
.\website\.venv\Scripts\python.exe -X utf8 website\test_website.py --production --base-url https://YOUR-SERVICE.onrender.com
```

Warm the service by opening it first. Free services sleep after 15 minutes without traffic and can take about a minute to wake. The hosted UI provides retry guidance for this; running a local Windows launcher cannot restart Render. Render recommends Free instances for demos and testing rather than production applications requiring reliable availability. [Free service limitations](https://render.com/docs/free)

### Settings, updates and failures

`render.yaml` is the source of truth for build/start commands, health checks and environment variables. `.python-version` selects Python 3.12. The dependency files preserve the saved model's sklearn/NumPy versions. The start command binds `0.0.0.0:$PORT`, uses one model worker with four request threads, and periodically recycles the worker. The app rejects uploads over 15 MB, serializes decoding/plotting/inference, keeps errors as JSON, and does not save uploaded photographs.

If using **New → Web Service** manually instead of the Blueprint, use the same settings below. Choose one creation method.

| Setting | Value |
|---|---|
| Repository / branch | `Proficient6/pepperLens` / `main` |
| Language | Python 3 |
| Root directory | Leave blank |
| Build command | `python -m pip install -r website/requirements-production.txt && python website/test_website.py --production` |
| Start command | Copy `startCommand` from `render.yaml` |
| Health check | `/api/health` |
| Environment | Copy `envVars` from `render.yaml`; do not override Python with a different version |
| Instance plan | Free for the first demo; reassess after measuring memory and availability |

| Failure | Check |
|---|---|
| `ModuleNotFoundError: website` or shared DSP module missing | Root directory must be blank; start from the repository root. |
| Missing model or Git LFS pointer error | `website/trained_run/pepper_models.joblib` must be the actual saved model. |
| Package version/wheel installation failure | Inspect the first failed package in build logs and the selected Python version. Preserve sklearn/NumPy compatibility when resolving it. |
| Production smoke test fails | Read the assertion/error in build logs; do not remove the test to hide a failure. |
| 502, worker killed, out of memory | Inspect logs and memory metrics; reduce input size and assess instance resources. |
| Busy / HTTP 503 JSON | Another image is processing; retry after it finishes. |
| Public app works but local app is offline | They are separate servers; use the launcher only for localhost. |

Future pushes to the connected branch deploy automatically unless automatic deploys are disabled. Inspect each deploy's logs and repeat one upload after release. Roll back through Render if a new version fails. The Colab notebook remains the training workspace; Render serves the saved model. Public availability and Linux compatibility are only verified after a successful real deployment and public upload test.
