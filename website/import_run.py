"""Copy only expected data artifacts from a user-produced training ZIP."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ALLOWED = {'.json', '.csv', '.png', '.joblib'}


def import_run(archive_path, destination=None):
    destination = Path(destination) if destination else ROOT / 'trained_run'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            parts = info.filename.split('/')
            if len(parts) == 2 and parts[0] == 'artifacts_all_models':
                name = parts[1]
                if not name or Path(name).name != name or '\\' in name or Path(name).suffix not in ALLOWED:
                    raise ValueError('Unexpected artifact filename in training ZIP.')
                if info.file_size > 200*1024*1024:
                    raise ValueError('Unexpected artifact size in training ZIP.')
                target = destination / name
                with archive.open(info) as source, target.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output)
        for name in ['dataset_source.json', 'source_versions.json']:
            if name in archive.namelist():
                (destination / name).write_bytes(archive.read(name))
        # Attached Python is inspected as text; it is never executed by this importer.
        if 'pepper_model.py' in archive.namelist():
            attached = ast.parse(archive.read('pepper_model.py').decode('utf-8'))
            current = ast.parse((ROOT.parent / 'pepper_model.py').read_text(encoding='utf-8'))
            definitions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                                        for node in tree.body if isinstance(node, ast.FunctionDef)}
            actual, expected = definitions(current), definitions(attached)
            for name in ['read_rgb', 'preprocess', 'extract_features', 'predict_all']:
                if actual[name] != expected[name]:
                    raise ValueError(f'The saved run uses different {name} code. Review feature compatibility.')
    for required in ['pepper_models.joblib', 'metrics.json', 'model_comparison.csv', 'environment.json']:
        if not (destination / required).is_file():
            raise ValueError(f'Training ZIP is missing {required}.')
    provenance = {'input_archive': Path(archive_path).name,
                  'input_archive_sha256': hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
                  'source_feature_compatibility': 'AST matched against current shared feature/inference code'}
    (destination / 'import_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    metrics = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
    print('Imported:', destination)
    print('Selected classifier:', metrics['selected_model'])
    print('Recorded test accuracy:', f"{metrics['test_accuracy_counting_rejections_as_errors']:.4%}")
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    import_run(args.archive)
