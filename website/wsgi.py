"""Production HTTP entry point; reuses the saved models and DSP implementation."""
import io
import json
import logging
import os
from pathlib import Path
import threading
import urllib.parse

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from flask import Flask, Response, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from website.server import MAX_UPLOAD, PLOTS, ROOT, PepperLens, plt


def create_app(service=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD, JSON_SORT_KEYS=False)
    app.json.sort_keys = False
    # Render terminates HTTPS at its reverse proxy; never trust forwarded Host.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=1, x_host=0)
    service = service or PepperLens(Path(os.environ.get('PEPPERLENS_RUN_DIR', ROOT / 'trained_run')))
    app.extensions['pepperlens'] = service
    prediction_slot = threading.BoundedSemaphore(1)
    app.extensions['prediction_slot'] = prediction_slot

    def json_response(payload, status=200):
        # Reject non-finite outputs instead of emitting invalid JSON.
        return Response(json.dumps(payload, allow_nan=False), status=status,
                        content_type='application/json; charset=utf-8')

    @app.after_request
    def response_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'self'; form-action 'self'"
        )
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/health')
    def health():
        return json_response({'application': 'PepperLens', 'status': 'ready',
                              'selected_model': service.bundle['selected_model'],
                              'mode': 'production', 'revision': os.environ.get('RENDER_GIT_COMMIT', 'local')})

    @app.get('/api/training')
    def training():
        return json_response(service.report)

    @app.get('/api/reports.zip')
    def reports_zip():
        return send_file(io.BytesIO(service.report_zip()), mimetype='application/zip',
                         as_attachment=True, download_name='PepperLens_training_reports.zip')

    @app.get('/reports/<path:name>')
    def report_file(name):
        allowed = set(PLOTS) | set(service.report['report_files'])
        path = service.run_dir / name
        if name not in allowed or not path.is_file():
            return json_response({'error': 'Report not found.'}, 404)
        return send_file(path, as_attachment=not name.endswith('.png'), download_name=name)

    @app.post('/api/predict')
    def predict():
        origin = request.headers.get('Origin')
        if origin and urllib.parse.urlsplit(origin).netloc.lower() != request.host.lower():
            return json_response({'error': 'Use the upload form on this website.'}, 403)
        length = request.content_length
        if length is not None and not 0 < length <= MAX_UPLOAD:
            return json_response({'error': 'Choose an image up to 15 MB.'}, 413)
        # Keep image decoding and plotting bounded as well as model inference.
        # Health/report requests remain responsive while one prediction runs.
        if not prediction_slot.acquire(blocking=False):
            response = json_response({'error': 'Another leaf is being analyzed. Wait a few seconds and retry.'}, 503)
            response.headers['Retry-After'] = '5'
            return response
        try:
            content = request.get_data(cache=False)
            if not content:
                return json_response({'error': 'Choose an image up to 15 MB.'}, 413)
            return json_response(service.analyze(content))
        except ValueError as error:
            return json_response({'error': str(error)}, 422)
        finally:
            plt.close('all')
            prediction_slot.release()

    @app.get('/')
    def index():
        return send_file(ROOT / 'static' / 'index.html', max_age=0)

    @app.get('/<name>')
    def static_asset(name):
        if name not in {'style.css', 'app.js', 'favicon.svg'}:
            return json_response({'error': 'Page not found.'}, 404)
        return send_file(ROOT / 'static' / name, max_age=0)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        return json_response({'error': 'Choose an image up to 15 MB.'}, 413)

    @app.errorhandler(HTTPException)
    def http_error(error):
        return json_response({'error': error.description}, error.code)

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception('PepperLens request failed')
        return json_response({'error': 'The image could not be analyzed. Please retry; the server log has details.'}, 500)

    app.logger.setLevel(logging.INFO)
    return app


app = create_app()
