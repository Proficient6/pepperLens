"""Shared RGB/HSV features, grouped evaluation, and seven classical classifiers."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import cv2
import joblib
import mahotas
import numpy as np
import pandas as pd
import sklearn
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

VERSION = 'rgb-hsv-masked-532-v1'
SIZE = 256
LABELS = ['bacterial_spot', 'healthy']
MULTICLASS_LABELS = ['bacterial_spot', 'cercospora_leaf_spot', 'healthy', 'leaf_curl',
                     'nutrient_deficiency', 'powdery_mildew']
MODEL_NAMES = ['Logistic Regression', 'Linear Discriminant Analysis', 'K-Nearest Neighbors',
               'Decision Tree', 'Random Forest', 'Gaussian Naive Bayes', 'Support Vector Machine']
DISPLAY_LABELS = {'bacterial_spot': 'Bacterial spot', 'healthy': 'Healthy',
                  'cercospora_leaf_spot': 'Cercospora leaf spot', 'leaf_curl': 'Leaf curl',
                  'nutrient_deficiency': 'Nutrient deficiency', 'powdery_mildew': 'Powdery mildew'}

def display_label(label):
    return DISPLAY_LABELS.get(str(label), str(label).replace('_', ' ').title())

def condition_type(label):
    if str(label).lower() in {'healthy', 'healthy_leaf'}:
        return 'Healthy'
    if str(label).lower() in {'nutrient_deficiency', 'nutrition_deficiency', 'magnesium_deficiency'}:
        return 'Abiotic condition'
    return 'Disease / symptom class'

def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'))

def preprocess(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected an RGB uint8 image with three channels.')
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(rgb, (3, 3), 0.7)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    # Colour-based candidate leaf region. This is not a lesion annotation.
    green = cv2.inRange(hsv, (25, 35, 20), (100, 255, 255))
    brown = cv2.inRange(hsv, (5, 35, 15), (30, 255, 255))
    union = cv2.bitwise_or(green, brown)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(union.shape, np.uint8)
    if contours:
        # Filling the contour preserves dark / pale lesions inside the leaf.
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    coverage = float(np.count_nonzero(mask) / mask.size)
    if coverage < 0.01:
        raise ValueError('Too little usable leaf colour. Try a clear, close photo of one leaf.')
    return rgb, smooth, hsv, mask, coverage

def extract_features(rgb):
    rgb, smooth, hsv, mask, coverage = preprocess(rgb)
    histogram = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    histogram = histogram.ravel() / max(float(histogram.sum()), 1.0)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    # Reserve zero for background; use 32 gray levels to keep GLCMs cheap.
    quantized = np.where(mask > 0, gray // 8 + 1, 0).astype(np.uint8)
    texture = mahotas.features.haralick(quantized, ignore_zeros=True, return_mean=True)
    moments = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel()
    moments = -np.sign(moments) * np.log10(np.maximum(np.abs(moments), 1e-30))
    feature = np.concatenate([histogram, texture, moments]).astype(np.float32)
    if feature.shape != (532,) or not np.isfinite(feature).all():
        raise ValueError('Invalid feature vector; inspect the input image.')
    return feature

def make_manifest(root, output, groups_csv=None, class_names=None, conflict_policy='exclude'):
    """Index files without copying, renaming, or deleting any source image.

    groups_csv: path,group_id columns; paths relative to root or absolute.
    Group IDs must identify the same plant/source image across derivatives.
    Conflicting decoded duplicates are reported and all copies are excluded by
    default. 'error' writes the same audit, then stops for manual review.
    """
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if conflict_policy not in {'exclude', 'error'}:
        raise ValueError("conflict_policy must be 'exclude' or 'error'.")
    manual = {}
    if groups_csv:
        for r in pd.read_csv(groups_csv, dtype=str).itertuples():
            manual[str((root / r.path).resolve())] = r.group_id
    leaf_map_file = root / 'leaf-map.json'
    leaf_map = json.loads(leaf_map_file.read_text()) if leaf_map_file.exists() else {}
    upstream_classes = {'bacterial_spot': 'Pepper,_bell___Bacterial_spot',
                        'healthy': 'Pepper,_bell___healthy'}
    class_names = list(class_names) if class_names is not None else LABELS
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError('Provide at least two distinct class names.')
    rows, rejected = [], []
    for label in class_names:
        folder = root / label
        if not folder.is_dir():
            raise ValueError(f'Missing class folder: {folder}')
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            try:
                rgb = read_rgb(path)
                digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                tiny = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32))
                dct = cv2.dct(tiny.astype(np.float32))[:8, :8].ravel()[1:]
                phash = ''.join('1' if v > np.median(dct) else '0' for v in dct)
                identifier = path.stem.replace('_final_masked', '').split('___')[-1].split('copy')[0].strip().lower()
                suggestions = leaf_map.get(identifier, [])
                upstream_class = upstream_classes.get(label)
                matches = [s for s in suggestions if upstream_class and upstream_class in s]
                leaf_id = suggestions[0] if len(suggestions) == 1 else (matches[0] if len(matches) == 1 else None)
                source_group = manual.get(str(path), leaf_id or digest)
                basis = 'manual' if str(path) in manual else ('upstream_leaf_id' if leaf_id else 'image_hash_only')
                rows.append({'path': str(path), 'label': label, 'sha256': digest,
                             'phash': phash, 'group_id': source_group,
                             'source_group': source_group, 'group_basis': basis})
            except Exception as exc:
                rejected.append({'path': str(path), 'reason': str(exc)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No readable images found.')
    missing = set(class_names) - set(df.label)
    if missing:
        raise ValueError(f'No readable images in requested class(es): {sorted(missing)}')
    label_counts = df.groupby('sha256').label.nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)
    conflicts = df[df.sha256.isin(conflict_hashes)].copy()
    label_sets = df.groupby('sha256').label.agg(lambda values: ' | '.join(sorted(set(values))))
    conflicts['conflicting_labels'] = conflicts.sha256.map(label_sets)
    conflicts['action'] = 'excluded_from_training_and_evaluation' if conflict_policy == 'exclude' else 'requires_manual_review'
    conflicts.to_csv(output / 'label_conflicts.csv', index=False)
    pd.DataFrame(rejected, columns=['path', 'reason']).to_csv(output / 'unreadable.csv', index=False)
    df[df.duplicated('sha256', keep=False)].to_csv(output / 'exact_duplicates.csv', index=False)
    counts_before = {label: int(df.label.eq(label).sum()) for label in class_names}
    audit = {
        'conflict_policy': conflict_policy,
        'readable_files_before_cleanup': len(df), 'unreadable_files': len(rejected),
        'conflicting_image_groups': len(conflict_hashes), 'conflicting_files_detected': len(conflicts),
        'conflicting_files_excluded': len(conflicts) if conflict_policy == 'exclude' else 0,
        'counts_before_cleanup': counts_before,
    }
    if conflict_hashes and conflict_policy == 'error':
        audit['status'] = 'blocked_for_label_review'
        (output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        raise ValueError(f'Identical decoded images have conflicting labels: {len(conflicts)} files '
                         f'in {len(conflict_hashes)} groups. See {output / "label_conflicts.csv"}. '
                         "Use conflict_policy='exclude' to omit every copy without changing source files.")
    # Connect matching metadata groups, exact images, and exact perceptual hashes.
    # This is a conservative heuristic, NOT a complete near-duplicate detector.
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for column in ['group_id', 'sha256', 'phash']:
        first = {}
        for i, key in enumerate(df[column]):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    df['group_id'] = [f'group-{find(i)}' for i in range(len(df))]
    # Form groups BEFORE exclusion/deduplication so removed rows cannot break a
    # connection between surviving parent groups or transformed siblings.
    unambiguous = df[~df.sha256.isin(conflict_hashes)].copy()
    df = unambiguous.drop_duplicates('sha256').reset_index(drop=True)
    audit['same_label_duplicate_copies_removed'] = len(unambiguous) - len(df)
    audit['retained_unique_images'] = len(df)
    audit['counts_after_cleanup'] = {label: int(df.label.eq(label).sum()) for label in class_names}
    missing_after_cleanup = set(class_names) - set(df.label)
    audit['status'] = 'blocked_missing_classes_after_cleanup' if missing_after_cleanup else 'ready'
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    df.to_csv(output / 'manifest.csv', index=False)
    if missing_after_cleanup:
        raise ValueError('No usable images remain after label-conflict exclusion in class(es): '
                         f'{sorted(missing_after_cleanup)}. Review {output / "label_conflicts.csv"}; '
                         'add correctly labelled examples before training.')
    if conflict_hashes:
        print(f'[AUDIT] Excluded all {len(conflicts)} files in {len(conflict_hashes)} conflicting-label '
              'groups. Source images were preserved. See label_conflicts.csv in the output folder.', flush=True)
    print(f'[AUDIT] Retained {len(df)} unique unambiguous images. '
          f'Per class: {audit["counts_after_cleanup"]}', flush=True)
    df.attrs['data_audit'] = audit
    return df

def assign_splits(df, seed=42):
    df = df.copy()
    labels = set(df.label)
    if df.groupby('label').group_id.nunique().min() < 5:
        raise ValueError('Need at least five independent groups per class.')
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    df['split'] = 'train'
    for fold, (_, indices) in enumerate(folds.split(df, df.label, df.group_id)):
        if fold in (0, 1):
            df.loc[df.index[indices], 'split'] = 'test' if fold == 0 else 'validation'
    if df.groupby('group_id').split.nunique().max() != 1:
        raise AssertionError('A group crosses partitions.')
    for split in ['train', 'validation', 'test']:
        if set(df.loc[df.split == split, 'label']) != labels:
            raise ValueError(f'{split} lacks one class. Review group sizes before changing the seed.')
    return df

def cached_features(df, output):
    output = Path(output)
    cache = output / 'feature_cache' / VERSION
    cache.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(output / 'features.npy', mode='w+', dtype='float32', shape=(len(df), 532))
    failures, valid = [], np.ones(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples()):
        file = cache / (r.sha256 + '.npy')
        try:
            feature = None
            if file.exists():
                try:
                    candidate = np.load(file, allow_pickle=False)
                    if candidate.shape == (532,) and np.isfinite(candidate).all():
                        feature = candidate
                except (OSError, ValueError, EOFError):
                    pass  # interrupted/corrupt cache: recover from the unchanged source image
            if feature is None:
                feature = extract_features(read_rgb(r.path))
                temp = file.with_suffix('.tmp')
                with temp.open('wb') as handle:
                    np.save(handle, feature)
                temp.replace(file)
            matrix[i] = feature
        except Exception as exc:
            valid[i] = False
            matrix[i] = 0
            failures.append({'path': r.path, 'label': r.label, 'split': r.split, 'reason': str(exc)})
        if (i + 1) % 250 == 0:
            print(f'Features {i+1}/{len(df)}', flush=True)
    matrix.flush()
    pd.DataFrame(failures, columns=['path', 'label', 'split', 'reason']).to_csv(output / 'feature_failures.csv', index=False)
    return matrix, valid

def training_views(df, X, mask, output, target_rows=0):
    """Optional train-only synthetic views; never manufacture validation/test rows."""
    originals = df.loc[mask].reset_index(drop=True)
    if target_rows <= len(originals):
        return X[mask], originals.label.to_numpy()
    rng = np.random.default_rng(42)
    expanded = np.lib.format.open_memmap(Path(output) / 'training_views.npy', mode='w+',
                                        dtype='float32', shape=(target_rows, 532))
    expanded[:len(originals)] = X[mask]
    labels = originals.label.tolist()
    parents = originals.sha256.tolist()
    attempts = 0
    while len(labels) < target_rows:
        attempts += 1
        if attempts > target_rows * 3:
            raise RuntimeError('Too many augmentation failures. Review segmentation.')
        # Balance synthetic views, while retaining every original training image.
        available_labels = sorted(originals.label.unique())
        label = available_labels[len(labels) % len(available_labels)]
        pool = originals[originals.label == label]
        row = pool.iloc[int(rng.integers(len(pool)))]
        rgb = np.rot90(read_rgb(row.path), int(rng.integers(4)))
        if rng.random() < 0.5:
            rgb = np.fliplr(rgb)
        rgb = np.clip(rgb.astype(np.float32) * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        try:
            feature = extract_features(np.ascontiguousarray(rgb))
        except ValueError:
            continue
        expanded[len(labels)] = feature
        labels.append(label)
        parents.append(row.sha256)
        if len(labels) % 1000 == 0:
            expanded.flush()
            print(f'Training views {len(labels)}/{target_rows}', flush=True)
    expanded.flush()
    pd.DataFrame({'label': labels, 'parent_sha256': parents}).to_csv(Path(output) / 'training_views.csv', index=False)
    return expanded, np.asarray(labels)

def wilson(correct, total):
    if not total:
        return [None, None]
    z = 1.96
    p = correct / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    margin = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [float(center-margin), float(center+margin)]

def make_models(seed=42):
    """All seven original algorithm families; scaling is fitted on training only.

    LDA uses shrinkage to handle correlated/constant histogram columns. SVM
    deliberately has no internally fitted probability calibration; it votes using
    predict(), so ungrouped calibration folds cannot leak leaf identities.
    """
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight='balanced', random_state=seed)),
        'Linear Discriminant Analysis': make_pipeline(StandardScaler(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        'K-Nearest Neighbors': make_pipeline(StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree', n_jobs=2)),
        'Decision Tree': DecisionTreeClassifier(min_samples_leaf=2,
            class_weight='balanced', random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12,
            min_samples_leaf=2, class_weight='balanced', n_jobs=2, random_state=seed),
        'Gaussian Naive Bayes': make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-8)),
        'Support Vector Machine': make_pipeline(StandardScaler(),
            SVC(C=5, class_weight='balanced', random_state=seed)),
    }

def combine_votes(predictions, classes):
    """Require a strict majority. Multi-class plurality/ties are inconclusive."""
    if not predictions:
        raise ValueError('No classifier predictions available.')
    classes = list(classes)
    if len(set(classes)) != len(classes) or len(classes) < 2:
        raise ValueError('Invalid class list.')
    if any(p not in classes for p in predictions):
        raise ValueError('A classifier returned a class outside the bundle taxonomy.')
    counts = {label: predictions.count(label) for label in classes}
    maximum = max(counts.values())
    leaders = [label for label in classes if counts[label] == maximum]
    majority = len(leaders) == 1 and maximum > len(predictions) / 2
    return {'prediction': leaders[0] if majority else None, 'votes': counts,
            'votes_for_leading_class': maximum, 'total_models': len(predictions),
            'agreement': maximum / len(predictions), 'leading_classes': leaders,
            'status': 'majority' if majority else 'inconclusive',
            'method': 'equal hard voting, strict majority required; agreement is not a probability'}

def predict_all(bundle, feature):
    """One feature extraction, all fitted classifiers, class-aware overall vote."""
    if bundle.get('feature_version') != VERSION:
        raise ValueError('Feature version mismatch. Retrain with this feature code.')
    models = bundle.get('models')
    if models is None and 'model' in bundle:
        models = {'Random Forest': bundle['model']}
    if not models:
        raise ValueError('No trained classifiers found in the bundle.')
    feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    if feature.shape != (1, 532) or not np.isfinite(feature).all():
        raise ValueError('Expected a finite 532-feature vector.')
    classes = [str(c) for c in bundle['classes']]
    rows, predictions = [], []
    for name, model in models.items():
        if set(map(str, model.classes_)) != set(classes):
            raise ValueError(f'{name} was trained on different classes.')
        predicted = str(model.predict(feature)[0])
        score = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature)[0]
            score = float(probabilities[list(map(str, model.classes_)).index(predicted)])
        rows.append({'model': name, 'prediction': predicted,
                     'condition': display_label(predicted), 'condition_type': condition_type(predicted),
                     'model_score': score,
                     'score_note': 'uncalibrated class score' if score is not None else 'SVM: class vote only'})
        predictions.append(predicted)
    vote_comparison = combine_votes(predictions, classes)
    overall = dict(vote_comparison)
    selected_name = bundle.get('selected_model')
    if selected_name is not None:
        if selected_name not in models:
            raise ValueError('The validation-selected classifier is missing from the bundle.')
        selected_row = next(r for r in rows if r['model'] == selected_name)
        overall.update(prediction=selected_row['prediction'], status='selected_model',
                       selected_model=selected_name, model_score=selected_row['model_score'],
                       selection_metric=bundle.get('selection_metric'),
                       selection_score=bundle.get('selection_score'),
                       method='validation_selected_model',
                       votes_for_selected_class=vote_comparison['votes'][selected_row['prediction']])
    overall['seven_model_comparison'] = set(models) == set(MODEL_NAMES)
    if overall['prediction']:
        overall['condition'] = display_label(overall['prediction'])
        overall['condition_type'] = condition_type(overall['prediction'])
    else:
        overall['condition'] = 'Inconclusive — classifiers disagree'
        overall['condition_type'] = 'Undetermined'
    return {'per_model': rows, 'overall': overall, 'vote_comparison': vote_comparison}

def evaluate_predictions(actual, predicted, classes):
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(actual, predicted)),
        'classification_report': classification_report(actual, predicted, labels=classes,
            output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=classes).tolist(),
    }

def train(root, output='artifacts_all_models', groups_csv=None, target_train_views=0,
          class_names=None, feature_output=None, data_note=None, conflict_policy='exclude'):
    """Fit seven models and select the primary classifier on validation only.

    feature_output optionally reuses an existing feature cache without touching
    the previous model bundle. No classifier is selected using the test results.
    """
    output = Path(output)
    df = assign_splits(make_manifest(root, output, groups_csv, class_names, conflict_policy))
    classes = sorted(df.label.unique().tolist())
    df.to_csv(output / 'manifest.csv', index=False)
    if feature_output:
        feature_root = Path(feature_output)
        feature_root.mkdir(parents=True, exist_ok=True)
    else:
        feature_root = output
    X, valid = cached_features(df, feature_root)
    if feature_root != output:
        (output / 'feature_failures.csv').write_bytes((feature_root / 'feature_failures.csv').read_bytes())
    masks = {s: (df.split.eq(s).to_numpy() & valid) for s in ['train', 'validation', 'test']}
    for split, mask in masks.items():
        if set(df.loc[mask, 'label']) != set(classes):
            raise ValueError(f'{split} lacks a class after feature failures.')
    y = df.label.to_numpy()
    train_X, train_y = training_views(df, X, masks['train'], output, target_train_views)
    models = make_models()
    validation, test_predictions = {}, {}
    for name, model in models.items():
        print(f'Training {name} on {len(train_y)} rows...', flush=True)
        model.fit(train_X, train_y)
        val_pred = model.predict(X[masks['validation']])
        validation[name] = evaluate_predictions(y[masks['validation']], val_pred, classes)
    selected_name = max(models, key=lambda name: validation[name]['balanced_accuracy'])
    selection_score = validation[selected_name]['balanced_accuracy']
    # Freeze the choice before any test predictions. Ties follow declared model order.
    print(f'Selected on validation balanced accuracy: {selected_name} ({selection_score:.4f})', flush=True)
    test = masks['test']
    per_model_metrics = {}
    for name, model in models.items():
        prediction = model.predict(X[test])
        test_predictions[name] = prediction
        per_model_metrics[name] = evaluate_predictions(y[test], prediction, classes)
    votes = [combine_votes([str(test_predictions[name][i]) for name in models], classes)
             for i in range(int(test.sum()))]
    vote_predictions = np.array([r['prediction'] or '__inconclusive__' for r in votes])
    overall_predictions = np.asarray(test_predictions[selected_name])
    decided = vote_predictions != '__inconclusive__'
    correct = int((overall_predictions == y[test]).sum())
    n_total = int(df.split.eq('test').sum())
    report = {
        'feature_version': VERSION, 'bundle_version': 3, 'labels': classes,
        'selected_model': selected_name, 'selection_metric': 'validation_balanced_accuracy',
        'selection_score': float(selection_score),
        'primary_prediction_rule': 'Use the individual model selected on validation; no test-based selection',
        'model_names': list(models), 'ensemble_rule': 'Equal hard vote; strict majority required',
        'split_counts': pd.crosstab(df.split, df.label).to_dict(),
        'unique_images': len(df), 'feature_failures': int((~valid).sum()),
        'data_audit': df.attrs['data_audit'],
        'training_rows_including_synthetic_views': len(train_y),
        'grouping_basis_counts': df.group_basis.value_counts().to_dict(),
        'validation_per_model': validation, 'test_per_model': per_model_metrics,
        'test_total': n_total, 'test_accepted': int(test.sum()),
        'test_coverage': float(test.sum()/n_total),
        'ensemble_decided': int(decided.sum()),
        'ensemble_decision_coverage_on_accepted': float(decided.mean()),
        'test_accuracy_on_accepted': correct / int(test.sum()),
        'test_accuracy_counting_rejections_as_errors': correct / n_total,
        'ensemble_accuracy_on_decided': float(accuracy_score(y[test][decided], vote_predictions[decided])) if decided.any() else None,
        'ensemble_accuracy_counting_rejections_and_abstentions_as_errors': int((vote_predictions == y[test]).sum()) / n_total,
        'test_balanced_accuracy': per_model_metrics[selected_name]['balanced_accuracy'],
        'image_level_wilson_95': wilson(correct, int(test.sum())),
        'ci_note': 'Image-level interval; correlated photos reduce effective sample size.',
        'classification_report': classification_report(y[test], overall_predictions,
            labels=classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y[test], overall_predictions,
            labels=classes + ['__inconclusive__']).tolist(),
        'data_note': data_note or 'Dataset provenance and parent grouping require review.',
        'limitations': 'Internal comparison only. Hashes do not prove independence. Disease names are restricted to trained labels. Field accuracy is unmeasured.',
    }
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'opencv': cv2.__version__, 'sklearn': sklearn.__version__, 'mahotas': mahotas.__version__}
    bundle = {'models': models, 'bundle_version': 3, 'feature_version': VERSION,
              'selected_model': selected_name, 'selection_metric': report['selection_metric'],
              'selection_score': float(selection_score),
              'size': SIZE, 'classes': classes, 'versions': versions,
              'ensemble_rule': report['ensemble_rule'], 'data_note': report['data_note']}
    joblib.dump(bundle, output / 'pepper_models.joblib')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    result = df.loc[test, ['path', 'label', 'group_id']].copy()
    for name, prediction in test_predictions.items():
        result[name] = prediction
    result['overall_prediction'] = overall_predictions
    result['vote_prediction'] = vote_predictions
    result['vote_agreement'] = [r['agreement'] for r in votes]
    result.to_csv(output / 'test_predictions.csv', index=False)
    table = pd.DataFrame([{'model': name, 'validation_balanced_accuracy': validation[name]['balanced_accuracy'],
                           'validation_macro_f1': validation[name]['classification_report']['macro avg']['f1-score'],
                           'test_accuracy': metrics['accuracy'], 'test_balanced_accuracy': metrics['balanced_accuracy'],
                           'test_macro_f1': metrics['classification_report']['macro avg']['f1-score'],
                           'selected_for_main_prediction': name == selected_name}
                          for name, metrics in per_model_metrics.items()])
    table.to_csv(output / 'model_comparison.csv', index=False)
    print(table.to_string(index=False), flush=True)
    print(f'Primary selected model ({selected_name}): {correct}/{n_total} correct. '
          f'Optional voting decisions: {int(decided.sum())}/{int(test.sum())}', flush=True)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/plantvillage')
    parser.add_argument('--output', default='artifacts_all_models')
    parser.add_argument('--groups', default=None)
    parser.add_argument('--train-views', type=int, default=0, help='Optional total training views; synthetic, not independent images')
    parser.add_argument('--classes', nargs='+', default=None, help='Class folder names; defaults to bacterial_spot healthy')
    parser.add_argument('--feature-output', default=None, help='Existing cache to reuse without altering the previous trained bundle')
    parser.add_argument('--data-note', default=None)
    parser.add_argument('--conflict-policy', choices=['exclude', 'error'], default='exclude')
    args = parser.parse_args()
    train(args.data, args.output, args.groups, args.train_views, args.classes, args.feature_output, args.data_note, args.conflict_policy)
