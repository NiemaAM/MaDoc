"""The web application of MaDoc (FastAPI): see server.py.

    config.py    the settings (device, CPU threads, options of the run), kept in a JSON file
    store.py     the documents of the application: one folder per document (scans, lines, record)
    pagexml.py   the lines of a page written as PageXML (what kraken reads to recognize them)
    pageone.py   page 1 of an act (names and marginal notes): its fields, compared with page 2
    process.py   the steps of a run: segmentation, recognition, formatting, translation, spelling
    jobs.py      one worker thread: the steps run one at a time (one GPU), their state is kept in the document
    monitor.py   CPU / memory / GPU samples and the durations of the steps (Monitoring page, /metrics)
    api.py       the HTTP routes
    static/      the interface (one page, no build step: Alpine.js is in static/vendor)
"""
