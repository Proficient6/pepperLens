"""Coursework plots of the shared feature pipeline; FFT/Sobel are diagnostics only."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pepper_model import preprocess, extract_features


def plot_processing_stages(image):
    """Show channel order, colour coordinates, nonlinear masks and segmentation."""
    rgb, smooth, hsv, mask, coverage = preprocess(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restored = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(restored, rgb)
    green = cv2.inRange(hsv, (25, 35, 20), (100, 255, 255))
    brown = cv2.inRange(hsv, (5, 35, 15), (30, 255, 255))
    union = cv2.bitwise_or(green, brown)
    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    segmented = rgb.copy()
    segmented[mask == 0] = 0
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.hypot(gx, gy)
    log_spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    stages = [
        (rgb, '1. Resampled RGB: 256 x 256', None, None),
        (bgr, '2. BGR incorrectly shown as RGB\n(channel-order demonstration)', None, None),
        (restored, '3. Correct BGR -> RGB display', None, None),
        (smooth, '4. Gaussian smoothing\n3 x 3, sigma = 0.7 pixels', None, None),
        (hsv[:, :, 0], '5. Hue H: OpenCV range 0-179', 'hsv', (0, 179)),
        (hsv[:, :, 1], '6. Saturation S: 0-255', 'gray', (0, 255)),
        (hsv[:, :, 2], '7. Value V: 0-255', 'gray', (0, 255)),
        (gray, '8. Grayscale for texture', 'gray', (0, 255)),
        (green, '9. Green threshold mask', 'gray', (0, 255)),
        (brown, '10. Brown threshold mask', 'gray', (0, 255)),
        (union, '11. Mask union: bitwise OR', 'gray', (0, 255)),
        (closed, '12. Morphological closing', 'gray', (0, 255)),
        (mask, '13. Filled largest contour', 'gray', (0, 255)),
        (segmented, '14. Selected leaf pixels', None, None),
        (edge_magnitude, '15. Sobel edge magnitude\n(diagnostic only)', 'magma', None),
        (log_spectrum, '16. Log magnitude of 2-D DFT\n(diagnostic only)', 'magma', None),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for axis, (pixels, title, colour_map, limits) in zip(axes.flat, stages):
        options = {'cmap': colour_map} if colour_map else {}
        if limits:
            options.update(vmin=limits[0], vmax=limits[1])
        axis.imshow(pixels, **options)
        axis.set_title(title, fontsize=10)
        axis.axis('off')
    fig.suptitle(f'Actual leaf-processing pipeline; candidate mask coverage {coverage:.1%}', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    arrays = {'rgb': rgb, 'bgr': bgr, 'smooth': smooth, 'hsv': hsv, 'mask': mask,
              'green': green, 'brown': brown, 'closed': closed, 'segmented': segmented,
              'edge_magnitude': edge_magnitude, 'log_spectrum': log_spectrum}
    return fig, arrays


def plot_filter_analysis(image):
    """Relate the Gaussian convolution used by the model to its frequency response."""
    rgb, smooth, _, _, _ = preprocess(image)
    gray_before = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_after = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gaussian_1d = cv2.getGaussianKernel(3, .7)
    kernel = gaussian_1d @ gaussian_1d.T
    response = np.abs(np.fft.fftshift(np.fft.fft2(kernel, s=gray_before.shape)))
    frequency = np.fft.fftshift(np.fft.fftfreq(gray_before.shape[1]))
    before_spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_before))))
    after_spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_after))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(kernel, cmap='Greens')
    for i in range(3):
        for j in range(3):
            axes[0, 0].text(j, i, f'{kernel[i, j]:.3f}', ha='center', va='center')
    axes[0, 0].set_title('Normalized Gaussian convolution kernel')
    axes[0, 1].imshow(response, cmap='viridis', extent=(-.5, .5, -.5, .5), vmin=0, vmax=1)
    axes[0, 1].set_title('Magnitude frequency response |H(fx, fy)|')
    axes[0, 1].set_xlabel('Horizontal cycles / pixel')
    axes[0, 1].set_ylabel('Vertical cycles / pixel')
    axes[0, 2].plot(frequency, response[response.shape[0]//2])
    axes[0, 2].set_title('Filter response along fy = 0')
    axes[0, 2].set_xlabel('Horizontal cycles / pixel')
    axes[0, 2].set_ylabel('Amplitude gain')
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].grid(alpha=.3)
    shared_max = max(before_spectrum.max(), after_spectrum.max())
    axes[1, 0].imshow(before_spectrum, cmap='magma', vmin=0, vmax=shared_max)
    axes[1, 0].set_title('Log DFT magnitude before smoothing')
    axes[1, 1].imshow(after_spectrum, cmap='magma', vmin=0, vmax=shared_max)
    axes[1, 1].set_title('Log DFT magnitude after smoothing\n(same display scale)')
    residual = gray_before - gray_after
    limit = max(float(np.abs(residual).max()), 1)
    axes[1, 2].imshow(residual, cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[1, 2].set_title('Removed spatial detail: before - after')
    for axis in [axes[0, 0], *axes[1]]:
        axis.axis('off')
    fig.suptitle('DSP application: spatial low-pass filtering and its 2-D frequency response')
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig, kernel, response


def plot_feature_summary(image):
    """Visualize the same 532 values supplied to every classifier."""
    feature = extract_features(image)
    histogram = feature[:512].reshape(8, 8, 8)
    texture = feature[512:525]
    hu = feature[525:532]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, marginal, title, upper in zip(
            axes, [histogram.sum(axis=(1, 2)), histogram.sum(axis=(0, 2)), histogram.sum(axis=(0, 1))],
            ['Hue marginal', 'Saturation marginal', 'Value marginal'], [180, 256, 256]):
        centers = (np.arange(8) + .5) * upper / 8
        axis.bar(centers, marginal, width=upper/10, color='#36734b')
        axis.set_title(title + ' of masked 8 x 8 x 8 histogram')
        axis.set_xlabel('Channel-bin center')
        axis.set_ylabel('Fraction of selected pixels')
        axis.set_xlim(0, upper)
    fig.tight_layout()
    texture_table = pd.DataFrame({'Descriptor': [f'Haralick {i+1}' for i in range(13)],
                                  'Actual feature value': texture})
    hu_table = pd.DataFrame({'Descriptor': [f'Log-transformed Hu {i+1}' for i in range(7)],
                             'Actual feature value': hu})
    return fig, texture_table, hu_table, feature
