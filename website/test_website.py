"""Check the saved model without a local dataset; optionally use your own image."""
import argparse
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from server import PepperLens
from pepper_model import extract_features, read_rgb


def check_http(base, photo, expected, saved_report):
    def request(path, content=None, headers=None):
        return urllib.request.urlopen(urllib.request.Request(base+path, data=content, headers=headers or {}), timeout=90)
    with request('/api/health') as response:
        assert json.load(response)['status'] == 'ready'
    with request('/') as response:
        assert response.status == 200 and b'Leaf lab' in response.read()
    with request('/api/training') as response:
        report = json.load(response)
        assert report['test_correct'] == saved_report['test_correct'] and len(report['comparison']) == 7
    with request('/reports/dsp_processing_stages.png') as response:
        assert response.read(8) == b'\x89PNG\r\n\x1a\n'
    with request('/api/predict', photo, {'Content-Type': 'application/octet-stream'}) as response:
        result = json.load(response)
        assert result['overall']['prediction'] == expected
        assert len(result['per_model']) == 7 and len(result['stages']) == 15
    for path, body, headers, status in [
        ('/api/predict', b'not an image', {}, 422),
        ('/api/predict', b'', {}, 413),
        ('/api/predict', b'image', {'Origin': 'https://example.invalid'}, 403),
        ('/reports/../pepper_models.joblib', None, {}, 404),
        ('/reports/pepper_models.joblib', None, {}, 404),
        ('/api/missing', None, {}, 404),
    ]:
        try:
            request(path, body, headers)
        except urllib.error.HTTPError as error:
            assert error.code == status, (path, error.code)
        else:
            raise AssertionError(f'{path} should return {status}')
    with request('/api/reports.zip') as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            assert archive.testzip() is None
            assert 'metrics.json' in archive.namelist()
            assert not any(name.endswith('.joblib') for name in archive.namelist())


def check_production(app, photo, expected, saved_report):
    """Exercise the real WSGI routes and bounded concurrency without opening a port."""
    from unittest.mock import patch
    from server import MAX_UPLOAD
    client = app.test_client()
    health = client.get('/api/health')
    assert health.json['mode'] == 'production' and health.json['status'] == 'ready'
    assert health.headers['Cache-Control'] == 'no-store'
    home = client.get('/')
    assert home.status_code == 200 and b'Leaf lab' in home.data
    assert "script-src 'self'" in home.headers['Content-Security-Policy']
    for name in ('app.js', 'style.css', 'favicon.svg'):
        assert client.get('/' + name).status_code == 200
    report = client.get('/api/training').json
    assert report['test_correct'] == saved_report['test_correct']
    assert len(report['comparison']) == 7
    assert client.get('/reports/dsp_processing_stages.png').data[:8] == b'\x89PNG\r\n\x1a\n'
    # Real image through the production route, including a same-origin HTTPS upload.
    response = client.post('/api/predict', data=photo, base_url='https://pepperlens.example',
                           headers={'Origin': 'https://pepperlens.example'})
    assert response.status_code == 200, response.json
    assert response.json['overall']['prediction'] == expected
    assert len(response.json['per_model']) == 7
    assert len(response.json['stages']) == 15 and len(response.json['figures']) == 3
    for path, body, headers, status in [
        ('/api/predict', b'not an image', {}, 422),
        ('/api/predict', b'', {}, 413),
        ('/api/predict', b'x' * (MAX_UPLOAD + 1), {}, 413),
        ('/api/predict', photo, {'Origin': 'https://elsewhere.invalid'}, 403),
    ]:
        response = client.post(path, data=body, headers=headers)
        assert response.status_code == status, (path, response.status_code)
        assert isinstance(response.json['error'], str)
    for path in ('/reports/../pepper_models.joblib', '/reports/pepper_models.joblib',
                 '/api/missing', '/wsgi.py'):
        assert client.get(path).status_code == 404
    assert client.get('/api/predict').status_code == 405
    # During inference the health endpoint must stay responsive; no second decode starts.
    slot = app.extensions['prediction_slot']
    assert slot.acquire(blocking=False)
    try:
        assert client.get('/api/health').status_code == 200
        response = client.post('/api/predict', data=photo)
        assert response.status_code == 503 and response.headers['Retry-After'] == '5'
    finally:
        slot.release()
    service = app.extensions['pepperlens']
    with patch.object(service, 'analyze', side_effect=RuntimeError('private failure detail')), \
         patch.object(app.logger, 'exception') as log_error:
        response = client.post('/api/predict', data=photo)
        assert response.status_code == 500
        assert b'private failure detail' not in response.data
        log_error.assert_called_once()
    assert slot.acquire(blocking=False), 'Failed prediction must release the slot.'
    slot.release()
    assert client.get('/api/health').status_code == 200
    with zipfile.ZipFile(io.BytesIO(client.get('/api/reports.zip').data)) as archive:
        assert archive.testzip() is None and 'metrics.json' in archive.namelist()
        assert not any(name.endswith('.joblib') for name in archive.namelist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url')
    parser.add_argument('--production', action='store_true', help='Check Flask routes, upload limits and concurrency.')
    parser.add_argument('--image', type=Path, help='Optional existing leaf photograph for a portable-copy test.')
    args = parser.parse_args()
    if args.production:
        sys.path.insert(0, str(ROOT.parent))
        from website.wsgi import app
        application = app.extensions['pepperlens']
    else:
        application = PepperLens(ROOT / 'trained_run')
    assert len(application.bundle['models']) == 7 and len(application.bundle['classes']) == 6
    assert application.bundle['selected_model'] == 'Random Forest'
    assert application.report['test_correct'] == round(application.metrics['test_accuracy_on_accepted'] * application.metrics['test_accepted'])
    if args.image:
        rgb = read_rgb(args.image)
        image_name = args.image.name
    else:
        # A generated leaf-like signal checks code integration, not disease accuracy.
        generated = Image.new('RGB', (256, 256), (238, 238, 238))
        draw = ImageDraw.Draw(generated)
        draw.ellipse((65, 28, 190, 225), fill=(44, 130, 36))
        draw.line((128, 215, 128, 44), fill=(72, 154, 47), width=3)
        draw.ellipse((100, 100, 118, 115), fill=(96, 58, 28))
        rgb = np.asarray(generated)
        image_name = 'generated_leaf_signal'
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    photo = buffer.getvalue()
    result = application.analyze(photo)
    selected = application.bundle['models']['Random Forest']
    feature = extract_features(rgb)
    expected = str(selected.predict(feature.reshape(1, -1))[0])
    assert result['overall']['prediction'] == expected
    assert np.isclose(result['overall']['model_score'], selected.predict_proba(feature.reshape(1, -1)).max())
    assert np.isclose(sum(row['score'] for row in result['distribution']), 1)
    assert result['features']['count'] == 532
    assert len(result['features']['histogram']) == 512
    assert len(result['features']['haralick']) == 13 and len(result['features']['hu']) == 7
    assert len(result['stages']) == 15 and len(result['figures']) == 3
    assert all(row['image'].startswith('data:image/png;base64,') for row in result['stages'])
    assert next(row for row in result['per_model'] if row['model']=='Support Vector Machine')['model_score'] is None
    for content in [b'bad image', None]:
        if content is None:
            blank = io.BytesIO()
            Image.new('RGB', (256, 256), 'white').save(blank, format='PNG')
            content = blank.getvalue()
        try:
            application.analyze(content)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/blank input should be rejected.')
    if args.base_url:
        check_http(args.base_url.rstrip('/'), photo, expected, application.report)
    if args.production:
        check_production(app, photo, expected, application.report)
    summary = {'status':'passed', 'actual_saved_models':True, 'input_image':image_name,
               'http_checks':bool(args.base_url), 'production_checks':args.production,
               'models':7, 'classes':6, 'new_image_stages':15,
               'new_image_figures':3, 'dataset_downloads':False, 'test_correct':application.report['test_correct'],
               'test_total':application.metrics['test_total'], 'test_accuracy':application.metrics['test_accuracy_counting_rejections_as_errors']}
    (ROOT / 'runtime').mkdir(exist_ok=True)
    (ROOT / 'runtime' / 'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
