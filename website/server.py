"""PepperLens local website: saved-run dashboard and real seven-model inference."""
import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
import zipfile

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import sklearn

from pepper_model import VERSION, display_label, extract_features, predict_all
from dsp_walkthrough import plot_processing_stages, plot_filter_analysis, plot_feature_summary

Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD = 15 * 1024 * 1024
PLOTS = {
    'model_results.png': ('Model comparison & confusion matrix', 'Saved during your Colab training run.'),
    'training_class_examples.png': ('Training examples by class', 'One retained training example per class.'),
    'dsp_processing_stages.png': ('Training image: 16 processing stages', 'Saved RGB/BGR, HSV, masks, smoothing, edges and DFT demonstration.'),
    'dsp_gaussian_frequency_response.png': ('Training image: Gaussian filter analysis', 'Spatial kernel, frequency response and before/after spectra.'),
    'dsp_colour_histograms.png': ('Training image: colour descriptors', 'Marginals of the masked 512-value HSV histogram.'),
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def as_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def figure_png(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    plt.close(figure)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def colour_image(values, cmap='magma'):
    values = np.asarray(values, dtype=float)
    scaled = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    return Image.fromarray((matplotlib.colormaps[cmap](scaled)[:, :, :3] * 255).astype(np.uint8))


class PepperLens:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.metrics = load_json(self.run_dir / 'metrics.json')
        self.environment = load_json(self.run_dir / 'environment.json')
        if sklearn.__version__ != self.environment['sklearn']:
            raise RuntimeError(f"This model requires scikit-learn {self.environment['sklearn']}; "
                               f"the interpreter has {sklearn.__version__}. Use website/.venv or Start-PepperLens.cmd.")
        self.bundle = joblib.load(self.run_dir / 'pepper_models.joblib')
        if self.bundle['feature_version'] != VERSION:
            raise ValueError('Saved model and image feature code do not match.')
        if self.bundle.get('selected_model') != self.metrics['selected_model']:
            raise ValueError('Saved model and training report disagree about the selected classifier.')
        if set(self.bundle['classes']) != set(self.metrics['labels']):
            raise ValueError('Saved model and training class lists differ.')
        self.lock = threading.Lock()
        self.report = self.prepare_report()

    def prepare_report(self):
        with (self.run_dir / 'model_comparison.csv').open(encoding='utf-8', newline='') as handle:
            comparison = list(csv.DictReader(handle))
        for row in comparison:
            for key in list(row):
                if key == 'selected_for_main_prediction':
                    row[key] = row[key].lower() == 'true'
                elif key != 'model':
                    row[key] = float(row[key])
        with (self.run_dir / 'test_predictions.csv').open(encoding='utf-8', newline='') as handle:
            predictions = list(csv.DictReader(handle))
        correct = sum(row['label'] == row['overall_prediction'] for row in predictions)
        accepted = self.metrics['test_accepted']
        if len(predictions) != accepted or not np.isclose(correct/self.metrics['test_total'],
                self.metrics['test_accuracy_counting_rejections_as_errors']):
            raise ValueError('Saved prediction rows do not reproduce the reported primary test accuracy.')
        reports = sorted(path.name for path in self.run_dir.iterdir()
                         if path.is_file() and path.suffix in {'.csv', '.json'})
        return {'metrics': self.metrics, 'environment': self.environment,
                'runtime': {'python': sys.version.split()[0], 'sklearn': sklearn.__version__,
                            'numpy': np.__version__, 'opencv': cv2.__version__},
                'comparison': comparison, 'test_correct': correct,
                'source': load_json(self.run_dir / 'dataset_source.json'),
                'plots': [{'filename': name, 'title': title, 'caption': caption,
                           'url': '/reports/' + name} for name, (title, caption) in PLOTS.items()
                          if (self.run_dir / name).is_file()],
                'report_files': reports,
                'score_note': 'Per-image confidence is an uncalibrated model score, not measured accuracy for that photograph.',
                'field_note': self.metrics['limitations']}

    def analyze(self, content):
        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(content)) as opened:
                if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError('Choose a photograph smaller than 25 megapixels.')
                original_size = [opened.width, opened.height]
                rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('This file is not a readable image. Try a JPG, PNG or WebP photograph.') from error
        with self.lock:
            feature = extract_features(rgb)
            result = predict_all(self.bundle, feature)
            primary = self.bundle['models'][self.bundle['selected_model']]
            probabilities = primary.predict_proba(feature.reshape(1, -1))[0] if hasattr(primary, 'predict_proba') else None
            distribution = [{'label': str(label), 'condition': display_label(label), 'score': float(value)}
                            for label, value in zip(primary.classes_, probabilities)] if probabilities is not None else []
            distribution.sort(key=lambda row: -row['score'])
            stage_figure, arrays = plot_processing_stages(rgb)
            filter_figure, kernel, _ = plot_filter_analysis(rgb)
            histogram_figure, texture, hu, demonstrated = plot_feature_summary(rgb)
            np.testing.assert_allclose(feature, demonstrated)
            tiles = [
                ('rgb', 'RGB input', 'Resampled to 256 × 256 pixels.', Image.fromarray(arrays['rgb'])),
                ('bgr', 'BGR channel-order demo', 'BGR values displayed as RGB to reveal swapped colours; not the classifier input.', Image.fromarray(arrays['bgr'])),
                ('smooth', 'Gaussian smoothing', '3 × 3 convolution; sigma 0.7 pixels.', Image.fromarray(arrays['smooth'])),
                ('hue', 'HSV · hue', 'OpenCV H values 0–179, shown in grayscale.', Image.fromarray(arrays['hsv'][:, :, 0])),
                ('saturation', 'HSV · saturation', 'S values 0–255.', Image.fromarray(arrays['hsv'][:, :, 1])),
                ('value', 'HSV · value', 'V values 0–255.', Image.fromarray(arrays['hsv'][:, :, 2])),
                ('gray', 'Grayscale', 'Intensity signal used for texture measurement.', Image.fromarray(cv2.cvtColor(arrays['smooth'], cv2.COLOR_RGB2GRAY))),
                ('green', 'Green mask', 'HSV threshold for green candidates.', Image.fromarray(arrays['green'])),
                ('brown', 'Brown mask', 'HSV threshold for brown candidates.', Image.fromarray(arrays['brown'])),
                ('union', 'Combined mask', 'Bitwise OR of green and brown candidates.', Image.fromarray(cv2.bitwise_or(arrays['green'], arrays['brown']))),
                ('closing', 'Morphological closing', '5 × 5 closing fills small gaps.', Image.fromarray(arrays['closed'])),
                ('mask', 'Filled leaf contour', 'Largest outer contour; candidate leaf area, not lesion area.', Image.fromarray(arrays['mask'])),
                ('segmented', 'Selected leaf pixels', 'Pixels used for masked colour descriptors.', Image.fromarray(arrays['segmented'])),
                ('sobel', 'Sobel edge magnitude', 'Spatial-gradient diagnostic; not a classifier input.', colour_image(arrays['edge_magnitude'])),
                ('dft', '2-D DFT spectrum', 'Log magnitude; bright centre represents low spatial frequencies. Diagnostic only.', colour_image(arrays['log_spectrum'])),
            ]
            result['stages'] = [{'id': key, 'title': title, 'caption': caption, 'image': as_png(image)}
                                for key, title, caption, image in tiles]
            result['figures'] = [
                {'title': 'Complete processing sheet', 'image': figure_png(stage_figure)},
                {'title': 'Gaussian kernel & frequency analysis', 'image': figure_png(filter_figure)},
                {'title': 'Masked colour histograms', 'image': figure_png(histogram_figure)},
            ]
            result['features'] = {'count': len(feature), 'histogram': feature[:512].tolist(),
                                  'haralick': feature[512:525].tolist(), 'hu': feature[525:].tolist(),
                                  'gaussian_kernel': kernel.tolist()}
            result['distribution'] = distribution
            result['mask_coverage'] = float(np.count_nonzero(arrays['mask']) / arrays['mask'].size)
            result['original_size'] = original_size
            result['processing_seconds'] = round(time.perf_counter() - started, 2)
            result['score_note'] = self.report['score_note']
            result['training_test_accuracy'] = self.metrics['test_accuracy_counting_rejections_as_errors']
            return result

    def report_zip(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in self.run_dir.iterdir():
                if path.is_file() and path.suffix in {'.json', '.csv', '.png'}:
                    archive.write(path, path.name)
        return buffer.getvalue()


def make_handler(application):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'PepperLens/1.0'

        def send(self, status, content, content_type, download=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store' if content_type.startswith('application/json') else 'no-cache')
            self.send_header('Connection', 'close')
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(content)

        def json(self, status, payload):
            self.send(status, json.dumps(payload, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8')

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == '/api/health':
                self.json(200, {'application': 'PepperLens', 'status': 'ready', 'selected_model': application.bundle['selected_model']})
            elif path == '/api/training':
                self.json(200, application.report)
            elif path == '/api/reports.zip':
                self.send(200, application.report_zip(), 'application/zip', 'PepperLens_training_reports.zip')
            elif path.startswith('/reports/'):
                name = urllib.parse.unquote(path[len('/reports/'):])
                allowed = set(PLOTS) | set(application.report['report_files'])
                if name not in allowed or not (application.run_dir / name).is_file():
                    self.json(404, {'error': 'Report not found.'})
                    return
                content_type = 'image/png' if name.endswith('.png') else 'application/json' if name.endswith('.json') else 'text/csv; charset=utf-8'
                self.send(200, (application.run_dir / name).read_bytes(), content_type,
                          name if not name.endswith('.png') else None)
            else:
                files = {'/': ('index.html', 'text/html; charset=utf-8'),
                         '/style.css': ('style.css', 'text/css; charset=utf-8'),
                         '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                         '/favicon.svg': ('favicon.svg', 'image/svg+xml')}
                if path not in files:
                    self.json(404, {'error': 'Page not found.'})
                    return
                filename, kind = files[path]
                self.send(200, (ROOT / 'static' / filename).read_bytes(), kind)

        def do_POST(self):
            if urllib.parse.urlsplit(self.path).path != '/api/predict':
                self.json(404, {'error': 'Endpoint not found.'})
                return
            origin = self.headers.get('Origin')
            if origin and urllib.parse.urlsplit(origin).netloc != self.headers.get('Host'):
                self.json(403, {'error': 'Use the upload form on this website.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD:
                self.json(413, {'error': 'Choose an image up to 15 MB.'})
                return
            try:
                result = application.analyze(self.rfile.read(length))
                self.json(200, result)
            except ValueError as error:
                self.json(422, {'error': str(error)})
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.json(500, {'error': 'The image could not be analyzed. Check the server terminal for details.'})

        def log_message(self, format, *args):
            print(self.log_date_time_string(), format % args, flush=True)

    return Handler


class LocalHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # No reverse-DNS lookup is needed for a loopback-only development server.
        TCPServer.server_bind(self)
        self.server_name = 'localhost'
        self.server_port = self.server_address[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the PepperLens website using the saved Colab model.')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'trained_run')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    application = PepperLens(args.run_dir)
    try:
        server = LocalHTTPServer(('127.0.0.1', args.port), make_handler(application))
    except OSError:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health', timeout=2) as response:
                existing = json.load(response)
            if existing.get('application') != 'PepperLens':
                raise ValueError('Another application is using this port.')
        except Exception:
            raise RuntimeError(f'Port {args.port} is already in use. Start with --port {args.port+1}.')
        print(f'PepperLens is already running: http://127.0.0.1:{args.port}', flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(f'http://127.0.0.1:{args.port}')
        sys.exit(0)
    server.daemon_threads = True
    print(f'PepperLens ready: http://127.0.0.1:{args.port}', flush=True)
    print('Saved Colab run loaded; no retraining or dataset download. Ctrl+C stops the server.', flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
