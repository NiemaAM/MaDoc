"""MaDoc, the web application (FastAPI): scans of a birth act -> lines -> text -> record.

    python src/server.py                       # http://127.0.0.1:7860
    python src/server.py --host 0.0.0.0 --device cuda:0
    MADOC_AUTH=ana:1234 python src/server.py   # a login is asked

The interface is a wizard in four steps: Pages (page 2 of the act and, when wanted, page 1; a file or the
camera), Lines (the lines found, which can be moved, deleted or drawn, then read again), Text (corrected by hand,
spelling check on demand) and Record (the fields, the validation, the downloads); plus Settings (device, CPU
threads, options) and Monitoring (CPU, memory, GPU, durations of the steps; /metrics for Prometheus).

One process only: the queue of the runs and the loaded models live in it (do not start several workers).
The package madoc_web holds the code; see its docstring.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Web application of the birth certificate pipeline (FastAPI).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", help="kraken device at the first start: auto, cpu, cuda:0 ... "
                                         "(the Settings page changes it afterwards)")
    parser.add_argument("--segmentation-model", type=Path)
    parser.add_argument("--recognition-model", type=Path)
    parser.add_argument("--data-dir", type=Path, help="where the documents and the settings are kept "
                                                      "(MADOC_DATA_DIR; default: a temporary folder)")
    parser.add_argument("--auth", metavar="USER:PASSWORD",
                        help="ask for a login (several: a:1,b:2); also MADOC_AUTH in .env, which keeps the password "
                             "out of the shell history")
    args = parser.parse_args()
    if args.data_dir:
        os.environ["MADOC_DATA_DIR"] = str(args.data_dir)
    if args.device:
        os.environ["MADOC_DEVICE"] = args.device
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    import uvicorn

    from madoc_web import api, process
    for kind, path in (("segmentation", args.segmentation_model), ("recognition", args.recognition_model)):
        if path:
            if not path.is_file():
                parser.error(f"model not found: {path}")
            process.MODELS[kind] = path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    auth = api.logins(args.auth or os.environ.get("MADOC_AUTH", ""))
    if args.host not in ("127.0.0.1", "localhost", "::1") and not auth:
        print("Warning: MaDoc listens on the network without a login (--auth / MADOC_AUTH): anyone who reaches "
              "this port can read the documents.", file=sys.stderr)
    uvicorn.run(api.create_app(auth), host=args.host, port=args.port, workers=1, log_level="info",
                access_log=False)     # the access log would hold nothing useful, and one line per poll


if __name__ == "__main__":
    main()
