#!/usr/bin/env python
"""Generate BlockRank ICR JSONL files from BEIR datasets with BM25 retrieval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


DATASETS = {
    "trec_covid": {"beir": "trec-covid", "split": "test"},
    "nfcorpus": {"beir": "nfcorpus", "split": "test"},
    "nq": {"beir": "nq", "split": "test"},
    "hotpotqa": {"beir": "hotpotqa", "split": "test"},
    "fiqa": {"beir": "fiqa", "split": "test"},
    "dbpedia_entity": {"beir": "dbpedia-entity", "split": "test"},
    "scidocs": {"beir": "scidocs", "split": "test"},
    "fever": {"beir": "fever", "split": "test"},
    "climate_fever": {"beir": "climate-fever", "split": "test"},
    "scifact": {"beir": "scifact", "split": "test"},
    "msmarco": {"beir": "msmarco", "split": "dev"},
}

for _meta in DATASETS.values():
    _meta["prebuilt"] = f"beir-v1.0.0-{_meta['beir']}.flat"
    _meta["topic"] = f"beir-v1.0.0-{_meta['beir']}-{_meta['split']}"

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _require_imports():
    try:
        from beir import util as beir_util  # noqa: F401
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install beir pyserini\n"
            "Pyserini also requires a working Java runtime."
        ) from exc


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_with_resume(url: str, out_path: Path, retries: int = 8, chunk_size: int = 1 << 20) -> None:
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        existing = out_path.stat().st_size if out_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        mode = "ab" if existing else "wb"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    return
                if existing and r.status_code != 206:
                    existing = 0
                    mode = "wb"
                r.raise_for_status()
                content_len = int(r.headers.get("content-length") or 0)
                total = existing + content_len if r.status_code == 206 else content_len or None
                with out_path.open(mode + "") as f, tqdm(
                    total=total,
                    initial=existing if r.status_code == 206 else 0,
                    unit="iB",
                    unit_scale=True,
                    desc=str(out_path),
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(60, 2**attempt))


def ensure_beir_dataset(raw_root: Path, beir_name: str) -> Path:
    dataset_dir = raw_root / beir_name
    if (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "queries.jsonl").exists():
        return dataset_dir

    raw_root.mkdir(parents=True, exist_ok=True)
    url = f"{BEIR_BASE_URL}/{beir_name}.zip"
    zip_path = raw_root / f"{beir_name}.zip"
    while True:
        download_with_resume(url, zip_path)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise zipfile.BadZipFile(f"Corrupt zip member: {bad_member}")
                zf.extractall(raw_root)
            break
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)
            download_with_resume(url, zip_path)
    return dataset_dir


def load_queries(dataset_dir: Path) -> dict[str, str]:
    return {str(row["_id"]): row["text"] for row in _read_jsonl(dataset_dir / "queries.jsonl")}


def load_qrels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    return load_qrels_tsv(qrels_path)


def load_qrels_tsv(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or (i == 0 and line.startswith("query-id")):
                continue
            qid, docid, score = line.split("\t")
            qrels.setdefault(str(qid), {})[str(docid)] = int(score)
    return qrels


def write_qrels(qrels: dict[str, dict[str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in qrels:
            for docid, score in qrels[qid].items():
                f.write(f"{qid}\t{docid}\t{score}\n")


def build_pyserini_collection(dataset_dir: Path, collection_dir: Path) -> None:
    done = collection_dir / ".complete"
    if done.exists():
        return

    collection_dir.mkdir(parents=True, exist_ok=True)
    out_file = collection_dir / "docs.jsonl"
    with (dataset_dir / "corpus.jsonl").open("r", encoding="utf-8") as src, out_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in tqdm(src, desc=f"Converting {dataset_dir.name} corpus"):
            row = json.loads(line)
            title = row.get("title") or ""
            text = row.get("text") or ""
            contents = f"{title}\n{text}".strip()
            dst.write(
                json.dumps(
                    {
                        "id": str(row["_id"]),
                        "contents": contents,
                        "title": title,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    done.touch()


def build_lucene_index(collection_dir: Path, index_dir: Path, threads: int) -> None:
    if (index_dir / "segments.gen").exists() or any(index_dir.glob("segments_*")):
        return

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storeRaw",
        "--storePositions",
        "--storeDocvectors",
    ]
    subprocess.run(cmd, check=True)


def retrieve_and_write(
    dataset_name: str,
    index_dir: Path,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    output_root: Path,
    topks: list[int],
    bm25_k1: float,
    bm25_b: float,
) -> None:
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=bm25_k1, b=bm25_b)
    retrieve_and_write_with_searcher(
        dataset_name=dataset_name,
        searcher=searcher,
        queries=queries,
        qrels=qrels,
        output_root=output_root,
        topks=topks,
    )


def retrieve_and_write_with_searcher(
    dataset_name: str,
    searcher,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    output_root: Path,
    topks: list[int],
) -> None:
    max_k = max(topks)

    writers = {}
    try:
        for topk in topks:
            out_dir = output_root / f"bm25-top{topk}"
            out_dir.mkdir(parents=True, exist_ok=True)
            writers[topk] = (out_dir / f"{dataset_name}.jsonl").open("w", encoding="utf-8")

        qids = [qid for qid in qrels.keys() if qid in queries]
        for qid in tqdm(qids, desc=f"Searching {dataset_name}"):
            hits = searcher.search(queries[qid], k=max_k)
            docs = []
            for hit in hits:
                raw = searcher.doc(hit.docid).raw()
                doc = json.loads(raw)
                docs.append(
                    {
                        "doc_id": str(hit.docid),
                        "title": doc.get("title") or "",
                        "text": doc.get("text") or "",
                    }
                )

            answer_ids = [docid for docid, rel in qrels[qid].items() if rel > 0]
            base = {
                "query": queries[qid],
                "query_id": str(qid),
                "answer_ids": answer_ids,
            }
            for topk, writer in writers.items():
                row = {**base, "documents": docs[:topk]}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for writer in writers.values():
            writer.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to generate.",
    )
    ap.add_argument("--topks", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    ap.add_argument("--output-root", type=Path, default=Path("data/icr-beir-evals"))
    ap.add_argument("--raw-root", type=Path, default=Path("data/beir-raw"))
    ap.add_argument("--work-root", type=Path, default=Path("data/beir-bm25-work"))
    ap.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--bm25-k1", type=float, default=0.9)
    ap.add_argument("--bm25-b", type=float, default=0.4)
    ap.add_argument(
        "--java-home",
        type=Path,
        default=Path("/data/xuanlu/openjdk21") if Path("/data/xuanlu/openjdk21").exists() else None,
        help="JAVA_HOME for Pyserini. Defaults to /data/xuanlu/openjdk21 when present.",
    )
    ap.add_argument("--skip-index", action="store_true", help="Assume Lucene indexes already exist.")
    ap.add_argument(
        "--prebuilt",
        action="store_true",
        help="Use Pyserini prebuilt BEIR indexes and local qrels instead of downloading raw BEIR data.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.java_home is not None:
        java_home = str(args.java_home)
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
        for candidate in (
            Path(java_home) / "lib" / "server" / "libjvm.so",
            Path(java_home) / "lib" / "jvm" / "lib" / "server" / "libjvm.so",
            Path(java_home) / "jre" / "lib" / "amd64" / "server" / "libjvm.so",
        ):
            if candidate.exists():
                os.environ["JVM_PATH"] = str(candidate)
                break
    _require_imports()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "qrels").mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        meta = DATASETS[dataset_name]
        beir_name = meta["beir"]
        split = meta["split"]

        if args.prebuilt:
            from pyserini.search import get_topics
            from pyserini.search.lucene import LuceneSearcher

            qrels = load_qrels_tsv(args.output_root / "qrels" / f"{dataset_name}.tsv")
            raw_topics = get_topics(meta["topic"])
            queries = {
                str(qid): (topic.get("title") or topic.get("text") or str(topic))
                for qid, topic in raw_topics.items()
            }
            searcher = LuceneSearcher.from_prebuilt_index(meta["prebuilt"])
            searcher.set_bm25(k1=args.bm25_k1, b=args.bm25_b)
            retrieve_and_write_with_searcher(
                dataset_name=dataset_name,
                searcher=searcher,
                queries=queries,
                qrels=qrels,
                output_root=args.output_root,
                topks=sorted(set(args.topks)),
            )
            continue

        dataset_dir = ensure_beir_dataset(args.raw_root, beir_name)
        queries = load_queries(dataset_dir)
        qrels = load_qrels(dataset_dir, split)
        write_qrels(qrels, args.output_root / "qrels" / f"{dataset_name}.tsv")

        collection_dir = args.work_root / "collections" / beir_name
        index_dir = args.work_root / "indexes" / beir_name
        if not args.skip_index:
            build_pyserini_collection(dataset_dir, collection_dir)
            build_lucene_index(collection_dir, index_dir, args.threads)

        retrieve_and_write(
            dataset_name=dataset_name,
            index_dir=index_dir,
            queries=queries,
            qrels=qrels,
            output_root=args.output_root,
            topks=sorted(set(args.topks)),
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )


if __name__ == "__main__":
    main()
