# Day 22 — LangSmith + Prompt Versioning

**Học viên:** Nguyễn Tất Đạt · **MSSV:** 2A202602578

[GitHub repository](https://github.com/Gaohonggg/K4-L3-DAY22-NguyenTatDat-2A202602578-LLMOpsPromptVersioning)

Bài lab xây dựng hệ thống hỏi đáp RAG có tracing, quản lý prompt qua LangSmith Hub,
routing A/B tất định, đánh giá RAGAS và hai custom Guardrails validators.

## Kết quả đã thực hiện

| Thành phần | Kết quả |
|---|---|
| RAG + tracing | FAISS 107 chunks; 50/50 câu chạy thành công |
| Prompt Hub + A/B | Push/pull hai prompt; 50/50 truy vấn, V1=19 và V2=31 |
| RAGAS | 50 QA mỗi phiên bản; đủ bốn metrics; cùng contexts cho mỗi cặp |
| Guardrails | PII demo 6/6; JSON demo 7/7 |
| Bằng chứng | Đủ bảy tệp bắt buộc trong [evidence/](evidence/) |

| Metric | V1 | V2 |
|---|---:|---:|
| Faithfulness | 0.9803 | 0.9378 |
| Answer relevancy | 0.9156 | 0.8525 |
| Context recall | 1.0000 | 1.0000 |
| Context precision | 0.9450 | 0.9417 |

Cả hai phiên bản đạt faithfulness ≥0.9. Báo cáo chi tiết, cấu hình và điểm từng mẫu:
[evidence/03_ragas_report.json](evidence/03_ragas_report.json).
Phân tích kết quả: [evidence/README.md](evidence/README.md).

LangSmith project: `day22-nguyentatdat-2a202602578`. 

## Thiết kế

- `config.py` tải `.env` và thiết lập tracing trước khi import LangChain.
- Bước 1 dùng LCEL: retriever → context/prompt → LLM → string parser.
  Các run con cho thấy documents và context thực tế đưa vào prompt.
- Bước 2 push/pull prompt bằng LangSmith Client, kiểm tra nội dung và commit hash.
  Routing dùng MD5 của `request_id`; prompt lỗi không được thay âm thầm bằng local.
- Bước 3 pull đúng commit Hub ghi ở bước 2. Reference chỉ dùng để đánh giá,
  không đưa vào prompt sinh câu trả lời. Retrieval dùng chung cho V1/V2.
- RAGAS đánh giá theo lô năm mẫu, lưu checkpoint sau mỗi lô thành công, từ chối
  điểm thiếu/NaN và tính trung bình trên đủ 50 mẫu mỗi phiên bản.
- Custom PII validator dùng regex; JSON validator sửa fences, nháy đơn và
  trailing comma bên ngoài chuỗi. Cả hai trả `FailResult(fix_value=...)`
  và nhận `OnFailAction.FIX` ở constructor.
- `run_all.py` dừng ở bước thất bại đầu tiên, trả exit code khác 0 khi có lỗi.

## Cấu trúc repo

`src/` giữ bốn entry point theo yêu cầu bài nộp, kèm `00_check_setup.py`,
`config.py`, `qa_pairs.py`, `run_all.py`.

Các module dùng chung trong `src/utils/`:
`llm_factory.py`, `data_loader.py`, `prompt_registry.py`,
`ragas_support.py`, `guardrails_validators.py`.

`tests/` kiểm tra routing, Hub loading, dataset/score integrity, validators và
trạng thái thoát của runner. `data/knowledge_base.txt` là nguồn tri thức của bài.
`evidence/` chứa bằng chứng nộp; `logs/` lưu output và checkpoint tại máy.

## Môi trường và cấu hình

Cấu hình đã chạy: Python 3.11.16, OpenAI `gpt-4o-mini`,
embeddings `text-embedding-3-small`, temperature 0,
chunk size 500 ký tự, overlap 50, top-k 3.

Phiên bản trong báo cáo: RAGAS 0.4.3, LangChain Core 1.6.7,
LangChain OpenAI 1.6.7, LangChain Community 0.3.31, FAISS CPU 1.15.1.
Guardrails AI 0.11.0 đã được kiểm tra ở bước chuẩn bị.
`requirements.lock` lưu bộ dependencies được resolve cho môi trường này.

Factory có hỗ trợ thêm providers khác, nhưng bài nộp và bước 3 được cấu hình,
kiểm tra với OpenAI. Chạy từng lệnh từ thư mục gốc repository; cần cài `uv` trước.

~~~bash
set -o pipefail
mkdir -p logs
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock 2>&1 | tee logs/00_install_dependencies.log
uv pip check --python .venv/bin/python 2>&1 | tee logs/00_dependency_check.log
cp -n .env.example .env
~~~

Điền `LANGCHAIN_API_KEY` và `OPENAI_API_KEY` trong `.env`, giữ `PROVIDER=openai`
và `LANGCHAIN_TRACING_V2=true`. Không đưa key vào terminal output hoặc Git.

~~~bash
.venv/bin/python src/config.py 2>&1 | tee logs/00_config_check.log
.venv/bin/python src/00_check_setup.py 2>&1 | tee logs/00_local_setup_check.log
.venv/bin/python src/00_check_setup.py --online 2>&1 | tee logs/00_online_setup_check.log
~~~

Chế độ `--online` gọi embeddings và LLM một lần mỗi loại, rồi gửi trace
`setup-check`; trace này không thay cho traces bắt buộc của bước 1/2.

## Kiểm tra và chạy bài lab

Tests dùng dữ liệu giả và mocks, không gọi LLM/API:

~~~bash
.venv/bin/python -m unittest discover -s tests -v 2>&1 | tee logs/final_tests.log
~~~

Chạy toàn bộ bốn bước:

~~~bash
.venv/bin/python -u src/run_all.py 2>&1 | tee logs/final_run_all.log
~~~

Chạy riêng từng bước khi cần:

~~~bash
.venv/bin/python -u src/run_all.py --step 1 2>&1 | tee logs/01_rag_full.log
.venv/bin/python -u src/run_all.py --step 2 2>&1 | tee logs/02_ab_routing_full.log evidence/02_ab_routing_log.txt
.venv/bin/python -u src/run_all.py --step 3 2>&1 | tee logs/03_ragas_full.log
.venv/bin/python -u src/04_guardrails_validator.py --demo pii 2>&1 | tee logs/04_pii_demo.log evidence/04_pii_demo_log.txt
.venv/bin/python -u src/04_guardrails_validator.py --demo json 2>&1 | tee logs/04_json_demo.log evidence/04_json_demo_log.txt
~~~

Bước 1/2 và RAGAS có phát sinh chi phí API. Chạy lại bước 1/2 tạo thêm traces và
tạo lại FAISS. Bước 3 cần `logs/02_prompt_manifest.json` từ bước 2;
khi clone repo mới, chạy bước 2 trước.

Chạy thử bằng `--limit 1` ở bước 1, `--limit 5` ở bước 2 và `--limit 2` ở bước 3.
Báo cáo smoke lưu riêng, không dùng để nộp.

## Checkpoint và evidence

`logs/03_ragas/` phân biệt cấu hình bằng fingerprint của dữ liệu, prompt commits,
model, retrieval và phiên bản thư viện. Chạy lại bước 3 cùng cấu hình sẽ tiếp tục
generations và lô đánh giá đã hoàn thành. Không chạy đồng thời hai tiến trình bước 3
cùng cấu hình vì chúng ghi cùng checkpoint.

Bước 3 xuất `data/ragas_report.json` và `evidence/03_ragas_report.json` khi hoàn
thành đủ mẫu. Không có checkpoint trên máy mới thì phải đánh giá lại. Output tại
máy được gitignore; bản report trong evidence được giữ để nộp.

Bằng chứng trace phải kiểm tra trên dashboard; thông báo flush queue trong
terminal không tự chứng minh đã có đủ traces hiển thị.

## Phạm vi và giới hạn

Điểm RAGAS là kết quả một lần đánh giá bằng LLM trên knowledge base của lab,
không phải cam kết chất lượng cho mọi dữ liệu. Context recall bằng 1 chỉ áp dụng
cho 50 reference của bộ QA này. Fact về model trong QA/knowledge base giữ theo
dữ liệu của bài, không phải tài liệu cập nhật thông số sản phẩm.

PII detector hỗ trợ các định dạng regex trong docstring, không bao phủ mọi loại
PII hoặc mọi định dạng điện thoại quốc tế. JSON fallback báo lỗi parse,
không tái tạo nội dung nghiệp vụ khi input không thể sửa.

## Tài liệu yêu cầu

- [CHECKPOINTS.md](CHECKPOINTS.md): hướng dẫn gốc của lab.
- [RUBRIC.md](RUBRIC.md): tiêu chí điểm bắt buộc và bonus.
- [SUBMISSION.md](SUBMISSION.md): cấu trúc bài nộp và deadline.
- [RULES.md](RULES.md): quy định làm bài và bảo mật.

Deadline trong tài liệu gốc: **23:59 ngày 08/10/2026, GMT+7**,
trừ khi giảng viên thông báo khác. Trước khi nộp, kiểm tra repo public,
đủ evidence, `.env` không bị track và không có API key trong file đã commit.
