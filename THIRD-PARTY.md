# Third-party licenses

docsearch is MIT ([LICENSE](LICENSE)). It uses the packages, models and programs below. Each has its own license.

How this list was made (2026-10-25):

- Python packages: the installed package metadata after `uv sync --frozen --extra cpu --extra ui` on macOS
  (Apple silicon), from `uv.lock`. The license is the text the package states (`License-Expression`, else
  `License`, else the trove classifier). Packages that install only on other platforms are listed as unverified.
- Models: the license metadata of the Hugging Face API (`https://huggingface.co/api/models/<id>`) and the
  license line of the model card, at the revision named (checked 2026-10-25). A pinned model uses that revision.
  A model that is not pinned downloads the latest revision, which can change its license: check it again
  when you download it.
- Search page: the `license` field in the npm registry (`https://registry.npmjs.org/<name>/<version>`) at the
  version in `ui/bun.lock`.
- Nothing here is legal advice. **Before corporate use, check each license and use policy yourself**, in
  particular every row marked unverified or restricted.

## Models of the default pipeline

These download to the Hugging Face cache on first use. None is in this repo.

| Model | Used for | License | Source |
| --- | --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | text vectors (`models.embedder`) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (API metadata and card, revision 97b0c614be4d; not pinned) |
| `Alibaba-NLP/gte-reranker-modernbert-base` | text reranker (`models.reranker`) | Apache-2.0 | https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base (API metadata and card, revision f7481e605550; not pinned) |
| `perplexity-ai/pplx-embed-v2-late-0.6b` | page-image vectors (`visual.default`) | MIT | https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b (card at the pinned revision 8fc2de24534a; API metadata at the latest revision 7bba8e7cf207) |
| `Qwen/Qwen3-VL-Reranker-2B` | page-image reranker on slides (`search.page_reranker`) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B (API metadata and card, revision 4bd860ac4f15; not pinned) |
| `docling-project/docling-layout-heron` | Docling page layout | Apache-2.0 | https://huggingface.co/docling-project/docling-layout-heron (API metadata and card, revision 8f39ad3c0b4c) |
| `docling-project/docling-models` | Docling table structure (TableFormer) | CDLA-Permissive-2.0 | https://huggingface.co/docling-project/docling-models (card at revision fc0f2d45e221; the latest revision 2199320848bb lists CDLA-Permissive-2.0 and Apache-2.0) |
| RapidOCR models (`PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx`, `ch_ppocr_mobile_v2.0_cls_mobile.onnx`) | OCR on images and scans | Apache-2.0 | the `rapidocr` package (3.9.2) downloads them from https://www.modelscope.cn/models/RapidAI/RapidOCR at tag v3.9.2; that repo states "Apache License 2.0". The models are ONNX conversions of PaddleOCR models, and https://github.com/PaddlePaddle/PaddleOCR is Apache-2.0. The `.onnx` files have no license file of their own. |

## Challenger models (off by default)

Only an experiment flag or a config change loads these. A challenger may carry terms beyond its license; the
row says so.

| Model | Alias | License | Source |
| --- | --- | --- | --- |
| `zeroentropy/zerank-2-reranker` | `zerank-2` | Apache-2.0 | https://huggingface.co/zeroentropy/zerank-2-reranker (API metadata, card and `LICENSE` file, revision 5eae30d5ee3c; not pinned) |
| `google/embeddinggemma-2` | `egemma2`, `egemma2-1120` | Apache-2.0; **restricted**: the Gemma Prohibited Use Policy also applies | https://huggingface.co/google/embeddinggemma-2 (API metadata and card, pinned revision 914f7f89142e). The card says "Deployments must adhere to the Gemma Prohibited Use Policy" (https://ai.google.dev/gemma/prohibited_use_policy). The policy limits use beyond Apache-2.0, so check it before corporate use. |
| `mixedbread-ai/mxbai-embed-large-v1` | (by env, M26) | Apache-2.0 | https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 (API metadata and card, revision b33106f585b9) |
| `Qwen/Qwen3-Reranker-0.6B` | `qwen3-0.6b` | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (API metadata and card, revision e61197ed4502; not pinned) |
| `Qwen/Qwen3-Reranker-4B` | `qwen3-4b` | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Reranker-4B (API metadata and card, revision 22e683669bc0; not pinned) |
| `BAAI/bge-reranker-v2-m3` | `bge-m3` | Apache-2.0 | https://huggingface.co/BAAI/bge-reranker-v2-m3 (API metadata and card, revision 953dc6f6f85a; not pinned) |
| `Qwen/Qwen3-VL-Reranker-8B` | `qwen3-vl-8b` | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B (API metadata and card, revision b212dc8c91a8; not pinned) |
| `perplexity-ai/pplx-embed-context-v1-0.6b` | `pplx-context` | MIT | https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b (API metadata and card, pinned revision b42df969d4d7); runs the model's own code |
| `tencent/EVIE-4.5B` | (never run; no config entry) | Apache-2.0; code and weights, third-party parts are Qwen3.5 (Apache-2.0) and colpali-engine (MIT) | https://huggingface.co/tencent/EVIE-4.5B (API metadata, card and `LICENSE` file, revision 498388f8f4c9; not pinned) |
| `lightonai/LateOn` | `lateon` | Apache-2.0 | https://huggingface.co/lightonai/LateOn (API metadata and card, pinned revision 62911e105059) |

## Models named in recorded trials only

`decisions.tsv` records trials of Nemotron 3.5 Embed and Parse, SlideOn, TableFormer V2 and LightOnOCR-3. No
config key or code path loads them by default, and their licenses are not verified in this file. Check each
license for corporate use before you add one as an alias, and add a row above.

## External programs (not bundled)

| Program | Used for | License |
| --- | --- | --- |
| LibreOffice (`soffice`) | `.doc` to `.docx`, Word and PowerPoint page renders | MPL-2.0 (https://www.libreoffice.org/about-us/licenses/; parts carry other open source licenses) |
| `textutil` (macOS) | `.doc` to `.docx` when LibreOffice is missing | part of macOS, under the macOS license; **unverified**: Apple publishes no separate license for this tool |
| Vendor CLIs and LLM APIs (`claude`, `codex`, `grok`, LiteLLM providers, Ollama) | optional LLM steps only | each vendor's terms; **unverified**: the terms differ per vendor and change, so check them for your account |

## Search page (`ui/`)

The built page bundles the runtime dependencies in `ui/package.json` and the two font packages. Versions come
from `ui/bun.lock`.

| Package | Version | License | Source |
| --- | --- | --- | --- |
| `react` | 19.3.1 | MIT | https://registry.npmjs.org/react/19.3.1 |
| `react-dom` | 19.3.1 | MIT | https://registry.npmjs.org/react-dom/19.3.1 |
| `scheduler` (a dependency of `react-dom`) | 0.28.0 | MIT | https://registry.npmjs.org/scheduler/0.28.0 |
| `@fontsource-variable/mulish` | 5.3.0 | OFL-1.1 | https://registry.npmjs.org/@fontsource-variable/mulish/5.3.0; the font itself: https://github.com/googlefonts/mulish (`OFL.txt`) |
| `@fontsource/charis-sil` | 5.3.0 | OFL-1.1 | https://registry.npmjs.org/@fontsource/charis-sil/5.3.0 |

The other packages in `ui/package.json` are build and test tools (Vite, Tailwind, Biome, Storybook,
Playwright). They run at build time, and the built page does not include their code [inferred].

## Python packages: direct dependencies

From `pyproject.toml` (`ui` extra included).

| Package | Version | License (as the package states it) | Metadata field |
| --- | --- | --- | --- |
| docling | 2.142.1 | MIT | License-Expression |
| huggingface-hub | 1.33.0 | Apache-2.0 (Apache Software License) | License |
| lancedb | 0.40.0 | Apache Software License | classifier |
| litellm | 1.104.2 | MIT | License-Expression |
| onnxruntime | 1.30.0 | MIT License (MIT License) | License |
| openpyxl | 3.1.5 | MIT (MIT License) | License |
| pandas | 3.0.6 | BSD License | classifier |
| pyarrow | 25.0.1 | Apache-2.0 | License-Expression |
| pyyaml | 6.0.3 | MIT (MIT License) | License |
| rapidocr | 3.9.2 | Apache-2.0 | License-Expression |
| rich | 15.0.0 | MIT (MIT License) | License |
| sentence-transformers | 6.1.0 | Apache-2.0 | License-Expression |
| tantivy | 0.26.2 | MIT | License-Expression |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | License-Expression |
| torchvision | 0.29.1 | BSD | License |
| django | 6.1.2 | BSD-3-Clause | License-Expression |
| django-ninja | 1.7.1 | MIT License | classifier |

## Python packages: everything else installed

The packages that the direct dependencies pull in (`ruff`, a dev tool, is left out).

| Package | Version | License (as the package states it) | Metadata field |
| --- | --- | --- | --- |
| accelerate | 1.15.0 | Apache (Apache Software License) | License |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 (Python Software Foundation License) | License |
| aiohttp | 3.14.4 | Apache-2.0 AND MIT | License-Expression |
| aiosignal | 1.4.0 | Apache 2.0 (Apache Software License) | License |
| annotated-doc | 0.0.5 | MIT | License-Expression |
| annotated-types | 0.8.0 | MIT | License-Expression |
| antlr4-python3-runtime | 4.9.3 | BSD | License |
| anyio | 4.15.1 | MIT | License-Expression |
| asgiref | 3.12.1 | BSD-3-Clause (BSD License) | License |
| attrs | 26.1.0 | MIT | License-Expression |
| beautifulsoup4 | 4.15.0 | MIT License (MIT License) | License |
| boto3 | 1.43.110 | Apache-2.0 | License |
| botocore | 1.43.110 | Apache-2.0 | License |
| certifi | 2026.10.19 | MPL-2.0 (Mozilla Public License 2.0 (MPL 2.0)) | License |
| charset-normalizer | 3.5.2 | MIT | License |
| click | 8.5.0 | BSD-3-Clause | License-Expression |
| cloudpickle | 3.1.2 | BSD-3-Clause (BSD License) | License |
| colorlog | 6.12.0 | MIT License (MIT License) | License |
| defusedxml | 0.7.1 | PSFL (Python Software Foundation License) | License |
| deprecation | 2.1.0 | Apache 2 (Apache Software License) | License |
| dill | 0.4.1 | BSD-3-Clause (BSD License) | License |
| distro | 1.9.0 | Apache License, Version 2.0 (Apache Software License) | License |
| doclang | 0.8.0 | Apache-2.0 | License-Expression |
| docling-core | 2.104.0 | MIT | License-Expression |
| docling-ibm-models | 4.0.4 | MIT | License-Expression |
| docling-parse | 7.23.0 | MIT | License-Expression |
| docling-slim | 2.142.1 | MIT | License-Expression |
| et-xmlfile | 2.0.0 | MIT (MIT License) | License |
| faker | 40.41.0 | MIT License (MIT License) | License |
| fastuuid | 0.14.0 | BSD License | classifier |
| filelock | 3.32.7 | MIT | License-Expression |
| filetype | 1.2.0 | MIT (MIT License) | License |
| flatbuffers | 25.12.19 | Apache 2.0 (Apache Software License) | License |
| frozenlist | 1.8.0 | Apache-2.0 | License |
| fsspec | 2026.9.0 | BSD-3-Clause | License-Expression |
| h11 | 0.16.0 | MIT (MIT License) | License |
| h2 | 4.4.1 | MIT | License-Expression |
| hf-xet | 1.7.0 | Apache-2.0 | License-Expression |
| hpack | 4.2.0 | MIT | License-Expression |
| httpcore | 1.0.9 | BSD-3-Clause | License-Expression |
| httpx | 0.28.1 | BSD-3-Clause (BSD License) | License |
| hyperframe | 6.1.0 | MIT License | classifier |
| idna | 3.20 | BSD-3-Clause | License-Expression |
| importlib-metadata | 8.9.0 | Apache-2.0 | License-Expression |
| jinja2 | 3.1.6 | BSD License | classifier |
| jiter | 0.17.0 | MIT | License-Expression |
| jmespath | 1.1.0 | MIT (MIT License) | License |
| joblib | 1.6.0 | BSD-3-Clause | License-Expression |
| jsonref | 1.1.0 | MIT | License |
| jsonschema | 4.26.0 | MIT | License-Expression |
| jsonschema-specifications | 2025.9.1 | MIT | License-Expression |
| lance-namespace | 0.13.0 | Apache-2.0 | License |
| lance-namespace-urllib3-client | 0.13.0 | Apache-2.0 | License |
| langcodes | 3.5.1 | MIT License | classifier |
| latex2mathml | 3.81.1 | MIT | License |
| lxml | 6.1.3 | BSD-3-Clause | License |
| mail-parser | 4.8.0 | Apache-2.0 | License-Expression |
| markdown-it-py | 4.2.0 | MIT License | classifier |
| marko | 2.2.4 | MIT | License |
| markupsafe | 3.0.4 | BSD-3-Clause | License-Expression |
| mdurl | 0.1.2 | MIT License | classifier |
| mpire | 2.10.2 | MIT (MIT License) | License |
| mpmath | 1.3.0 | BSD (BSD License) | License |
| multidict | 6.9.1 | Apache License 2.0 | License |
| multiprocess | 0.70.19 | BSD-3-Clause (BSD License) | License |
| narwhals | 2.26.0 | MIT | License-Expression |
| networkx | 3.7 | BSD-3-Clause | License-Expression |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | License-Expression |
| olefile | 0.47 | BSD (BSD License) | License |
| omegaconf | 2.3.1 | BSD License | classifier |
| openai | 2.54.0 | Apache-2.0 (Apache Software License) | License |
| opencv-python | 5.0.0.93 | Apache 2.0 (Apache Software License) | License |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | License-Expression |
| pillow | 13.0.0 | MIT-CMU | License-Expression |
| pluggy | 1.6.0 | MIT (MIT License) | License |
| polyfactory | 3.3.0 | MIT (MIT License) | License |
| propcache | 0.5.4 | Apache-2.0 | License |
| protobuf | 7.36.2 | 3-Clause BSD License | License |
| psutil | 7.2.2 | BSD-3-Clause | License |
| pyclipper | 1.4.0 | MIT (OSI Approved; MIT License) | License |
| pydantic | 2.14.0 | MIT | License-Expression |
| pydantic-core | 2.50.0 | MIT | License-Expression |
| pydantic-settings | 2.15.0 | MIT | License-Expression |
| pygments | 2.21.0 | BSD-2-Clause | License-Expression |
| pylatexenc | 2.11 | MIT (MIT License) | License |
| pypdfium2 | 5.14.0 | BSD-3-Clause, Apache-2.0, dependency licenses | License |
| python-dateutil | 2.9.0.post0 | Dual License (BSD License; Apache Software License) | License |
| python-docx | 1.2.0 | MIT (MIT License) | License |
| python-dotenv | 1.2.4 | BSD-3-Clause | License |
| python-oxmsg | 0.0.2 | MIT (MIT License) | License |
| python-pptx | 1.0.2 | MIT (MIT License) | License |
| referencing | 0.37.0 | MIT | License-Expression |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | License-Expression |
| requests | 2.34.2 | Apache-2.0 (Apache Software License) | License |
| rpds-py | 2026.9.1 | MIT | License-Expression |
| rtree | 1.4.1 | MIT | License-Expression |
| s3transfer | 0.19.2 | Apache License 2.0 (Apache Software License) | License |
| safetensors | 0.8.0 | Apache Software License | classifier |
| scikit-learn | 1.9.1 | BSD-3-Clause | License-Expression |
| scipy | 1.18.1 | BSD License | classifier |
| semchunk | 3.2.5 | MIT (MIT License) | License |
| setuptools | 84.2.1 | MIT | License-Expression |
| shapely | 2.2.0 | BSD-3-Clause | License-Expression |
| shellingham | 1.5.4 | ISC License (ISC License (ISCL)) | License |
| six | 1.17.0 | MIT (MIT License) | License |
| sniffio | 1.3.1 | MIT OR Apache-2.0 (MIT License; Apache Software License) | License |
| soupsieve | 2.10 | MIT | License-Expression |
| sqlparse | 0.6.0 | BSD License | classifier |
| sympy | 1.14.0 | BSD (BSD License) | License |
| tabulate | 0.10.0 | MIT | License-Expression |
| threadpoolctl | 3.7.0 | BSD-3-Clause | License-Expression |
| tiktoken | 0.14.0 | MIT License | License text |
| tokenizers | 0.23.2 | Apache Software License | classifier |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | License |
| transformers | 5.19.0 | Apache 2.0 License | License |
| tree-sitter | 0.26.0 | MIT License | classifier |
| tree-sitter-c | 0.24.2 | MIT (MIT License) | License |
| tree-sitter-javascript | 0.25.0 | MIT | License |
| tree-sitter-python | 0.25.0 | MIT | License |
| tree-sitter-typescript | 0.23.2 | MIT (MIT License) | License |
| typer | 0.26.8 | MIT | License-Expression |
| typing-extensions | 4.16.0 | PSF-2.0 | License-Expression |
| typing-inspection | 0.4.4 | MIT | License-Expression |
| urllib3 | 2.8.0 | MIT | License-Expression |
| websockets | 16.1.1 | BSD-3-Clause | License-Expression |
| xlsxwriter | 3.2.9 | BSD-2-Clause (BSD License) | License |
| yarl | 1.25.1 | Apache-2.0 | License |
| zipp | 4.1.1 | MIT | License-Expression |

## Python packages not installed here (unverified)

`uv.lock` lists these for other platforms or extras (Linux CUDA wheels, Windows). Their licenses are
**unverified**: `colorama`, `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit`, `nvidia-cublas`, `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`, `nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, `nvidia-nvtx`, `pywin32`, `triton`, `tzdata`. On the target machine, print each installed package and its stated license with:

```bash
uv run python -c "from importlib.metadata import distributions as ds; [print(d.metadata['Name'], d.version, d.metadata.get('License-Expression') or d.metadata.get('License')) for d in ds()]"
```

## Benchmark data

The datasets and their licenses are in README.md, "Data sources and licenses". The corpus files are not in this
repo.
