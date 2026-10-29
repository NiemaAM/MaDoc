"""kraken inside this process: the models are loaded once and kept (on the GPU when there is one).

The kraken command line imports torch, starts CUDA and reads its model again for every scan: seconds before any
work. Here the segmentation and recognition models stay loaded between scans, and the result is the same PageXML
the command line writes (`kraken -x ... segment -bl` / `kraken -f page -x ... ocr`), so the rest of MaDoc does not
change.

If kraken's library cannot be used (an older kraken, an error inside it), segmentation.py and recognition.py fall
back to the command line, as before. MADOC_KRAKEN_CLI=1 always uses the command line.
"""

import dataclasses
import os
import sys
import threading
import time
from pathlib import Path

_models = {}            # (kind, model path, device) -> loaded task model
_lock = threading.Lock()   # one scan at a time on the models (the GPU is shared by the users of the app)
# the models are let go after this many seconds without a scan: the GPU is shared with the trainings of the MLOps
# platform (MADOC_KRAKEN_KEEP_SECONDS, 0: kept for ever)
KEEP_SECONDS = int(os.environ.get("MADOC_KRAKEN_KEEP_SECONDS", "900"))
_last_use = [0.0]
_watcher = []


def enabled():
    """False when the command line must be used (MADOC_KRAKEN_CLI=1, or kraken's library missing)."""
    if os.environ.get("MADOC_KRAKEN_CLI") == "1":
        return False
    try:
        import kraken.tasks  # noqa: F401
        return True
    except ImportError:
        return False


def _device(device):
    """"cuda:0" / "cpu" -> kraken's (accelerator, devices), as the command line does (ketos.util.to_ptl_device)."""
    from kraken.ketos.util import to_ptl_device
    return to_ptl_device(device or "cpu")


def _free_gpu_cache():
    """Give back to the GPU the memory PyTorch kept after a page (a full page through the segmentation network
    reserves several GB). Kept, it leaves too little for Ollama's model, which then runs partly on the CPU and
    makes the spelling check very slow. The models themselves stay loaded (a few tens of MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def release_idle(now=None):
    """Let the models go when no scan used them for KEEP_SECONDS (the next scan loads them again) -> True when
    they were released."""
    now = time.monotonic() if now is None else now
    with _lock:
        if not _models or not KEEP_SECONDS or now - _last_use[0] < KEEP_SECONDS:
            return False
        _models.clear()
    _free_gpu_cache()
    print("kraken models released (no scan for a while)", file=sys.stderr)
    return True


def _watch():
    while True:
        time.sleep(60)
        release_idle()


def _model(kind, path, device):
    _last_use[0] = time.monotonic()
    if KEEP_SECONDS and not _watcher:
        _watcher.append(threading.Thread(target=_watch, daemon=True, name="kraken-models"))
        _watcher[0].start()
    key = (kind, str(Path(path).resolve()), device)
    if key not in _models:
        from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
        cls = SegmentationTaskModel if kind == "segmentation" else RecognitionTaskModel
        print(f"Loading the kraken {kind} model {Path(path).name} on {device} (kept for the next scans)",
              file=sys.stderr)
        _models[key] = cls.load_model(Path(path))
    return _models[key]


def _threads(config):
    """The thread limit of the command line (1 by default): the numbers of the CPU parts of kraken depend a little
    on the number of threads, and the same limit gives exactly the PageXML of the command line."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=config.num_threads)


def _serialize(results, image_size):
    from kraken import serialization
    return serialization.serialize(results, image_size=image_size, template="pagexml", template_source="native",
                                   processing_steps=[], sub_line_segmentation=True)


def segment(image_file, model, device="cpu"):
    """image -> PageXML text of the lines found (what `kraken -x -i image out.xml segment -bl -i model` writes)."""
    from kraken.configs import SegmentationInferenceConfig
    from kraken.lib.util import open_image

    accelerator, devices = _device(device)
    config = SegmentationInferenceConfig(accelerator=accelerator, device=devices)
    image = open_image(str(image_file))
    with _lock, _threads(config):
        try:
            result = _model("segmentation", model, device).predict(im=image, config=config)
        finally:
            _free_gpu_cache()
    result = dataclasses.replace(result, imagename=str(image_file))
    return _serialize(result, image.size)


def recognize(xml_file, model, device="cpu", pagexml=True):
    """PageXML of a segmentation (it names its image) -> PageXML with the text of every line (pagexml=True, what
    `kraken -f page -x ... ocr -m model` writes) or the text, one line per text line, in kraken's visual order."""
    from kraken.configs import RecognitionInferenceConfig
    from kraken.lib.util import open_image
    from kraken.lib.xml import XMLPage

    accelerator, devices = _device(device)
    # bidi_reordering=None is what `kraken ... ocr` uses without --base-dir (its ocr command replaces True by the
    # base direction given, None): the lines come out in the visual order that recognition.py turns back
    config = RecognitionInferenceConfig(accelerator=accelerator, device=devices, bidi_reordering=None)
    page = XMLPage(str(xml_file))
    segmentation = page.to_container()
    image = open_image(page.imagename)
    with _lock, _threads(config):
        try:
            predictions = list(_model("recognition", model, device).predict(im=image, segmentation=segmentation,
                                                                              config=config))
        finally:
            _free_gpu_cache()
    if not pagexml:
        return "\n".join(p.prediction for p in predictions)
    results = dataclasses.replace(segmentation, lines=predictions, imagename=page.imagename)
    return _serialize(results, image.size)


def run(step, function, fallback):
    """The step inside this process, or with the command line when kraken's library cannot be used."""
    if enabled():
        try:
            return function()
        except Exception as e:   # an API difference of another kraken version, a broken model...: the CLI says it
            print(f"kraken {step} in this process failed ({type(e).__name__}: {e}): using the command line",
                  file=sys.stderr)
    return fallback()