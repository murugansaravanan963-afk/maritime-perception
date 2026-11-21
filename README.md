# Maritime6 fresh dataset workspace

Status: **NOT RELEASED. No new training or HEF has been produced.**

This is the single source of truth for the replacement dataset. Start with [STATUS.md](STATUS.md). Prior v4/v5 data and checkpoints remain in `D:/Projects/maritime-perception`; none are overwritten.

The requested goal is six classes, 1,000+ distinct images per class, boat/shore priority, 90% mAP@0.50 as a measured goal, followed by Hailo-8L deployment. Public dataset sizes and structural validity are not guarantees of correct complete labels or of accuracy.

- [Source research and selection](sources/SOURCE_RESEARCH.md)
- [Approved class definitions](configs/taxonomy.json)
- [Annotation review and release procedure](sources/REVIEW_AND_RELEASE.md)
- [Prepared Sol handoff](sources/SOL_HANDOFF.md)
- [Hailo-8L deployment](sources/HAILO_DEPLOYMENT.md)
- `reports/`: native inventories, hashes, duplicate findings and visual sample findings.
- `raw/`: immutable publisher files; candidates only.
- `staging/`: merged unreleased candidates, explicit corrections and image-specific visual observations. Start with [the actual merge report](reports/MERGE_STATUS.md).
- `releases/`: populated only after release gates pass, currently empty.

## Reproduce the current audit on Windows

Use the existing Python environment; these audits need NumPy, Pillow, and SciPy (only SMD metadata). No GPU is needed.

```powershell
cd D:\Projects\maritime-dataset
& D:\Projects\maritime-perception\.venv\Scripts\python.exe scripts/acquire.py --profile research
& D:\Projects\maritime-perception\.venv\Scripts\python.exe scripts/acquire.py --profile surface_debris
& D:\Projects\maritime-perception\.venv\Scripts\python.exe scripts/audit_candidates.py debris
& D:\Projects\maritime-perception\.venv\Scripts\python.exe scripts/audit_candidates.py waterscenes
& D:\Projects\maritime-perception\.venv\Scripts\python.exe scripts/audit_candidates.py existing
& D:\Projects\maritime-perception\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Additional verified publisher acquisition profiles: `oostpoort` (~1.08 GB), `instream_wood_initial` (~1.05 GB). `instream_wood_all` is ~22 GB and is optional; avoid downloading the full collection until metadata/sample review supports it. Downloads resume `.part` files and verify the publisher MD5 when supplied. Read `configs/downloads.json` for exact URLs/sizes/checksums.

On Sol, use a **new** `/scratch/smurug25/maritime-dataset` directory. These same relative scripts work there after NumPy/Pillow installation; pass `--old /scratch/smurug25/maritime-perception` to `audit_candidates.py existing`. Do not submit a GPU training job for this workspace yet.

## Current review workflow

The native adapters are `scripts/stage_candidates.py`; they retain ambiguous objects, record source/image hashes and never create a training approval. `scripts/render_review.py` renders full frames and every supplied object crop. Actual observations are bound to the examined candidate, taxonomy, image and rendered evidence hashes in `staging/visual_observations.jsonl`. Explicit corrections preserve their unchanged parent/source/image provenance in `staging/corrections/`.

After adding a separately inspected review batch, run `python scripts/merge_candidates.py` to update the single merged inventory and readable coverage report. It rejects stale review/correction evidence and quarantines all members of conflicting exact duplicates. The visual observations are not a final split/dedup approval. Do not run the immutable first-session recording script again or generate bulk approvals from native class names/model confidence.

`openimages_people` candidates deliberately have no automatically mapped swimmer boxes. Swimwear/cap labels and generic person boxes do not establish that a person is in water. A native swimming-pool box is an exclusion signal; its absence is not proof of open water. Preserve all supplied co-occurring labels, the original metadata, author and license, and review the full frame.

The original Open Images Canoe and people pools overlap. Staging checks matching image bytes and full native annotations and keeps one canonical image record; overlapping photos do not increase coverage twice.
