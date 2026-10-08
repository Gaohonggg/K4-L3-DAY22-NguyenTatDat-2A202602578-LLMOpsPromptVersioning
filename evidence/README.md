# Bằng chứng và phân tích V1/V2

**Học viên:** Nguyễn Tất Đạt · **MSSV:** 2A202602578

## Bằng chứng

| File | Nội dung |
|---|---|
| [01_langsmith_traces.png](01_langsmith_traces.png) | Dashboard project `day22-nguyentatdat-2a202602578`, tổng 252 traces tại thời điểm chụp |
| [02_prompt_hub.png](02_prompt_hub.png) | Hai ChatPromptTemplate trên Hub; commit đầu `d2aeb8f8` và `e770ab1a` |
| [02_ab_routing_log.txt](02_ab_routing_log.txt) | Push/pull Hub, request ID, nhãn V1/V2, 50/50 thành công; routing 19/31 |
| [03_ragas_scores.png](03_ragas_scores.png) | Bảng bốn metrics trên 50 mẫu mỗi phiên bản |
| [03_ragas_report.json](03_ragas_report.json) | Điểm tổng hợp, 100 mẫu với contexts/answers/references/scores và cấu hình |
| [04_pii_demo_log.txt](04_pii_demo_log.txt) | Sáu case PII giả: đủ bốn loại, case hỗn hợp và case sạch |
| [04_json_demo_log.txt](04_json_demo_log.txt) | Bảy case JSON: hợp lệ, ba loại sửa lỗi, fallback và bảo toàn nội dung chuỗi |

Ảnh dashboard ghi tổng traces của project, gồm cả lượt chạy thử và các bước khác.
Không diễn giải số 252 thành số riêng của bước 1. Bước 1 chạy đủ 50 câu; bước 2
chạy 50 truy vấn A/B. Kiểm tra từng nhóm bằng bộ lọc `rag-query`/`step1` và
`ab-rag-query`/`step2` cùng metadata `batch_id`.


## Thiết kế thí nghiệm

V1 trả lời trực tiếp, ngắn gọn; V2 dùng Definition/Explanation và giọng chuyên gia.
Cả hai dùng context và nêu giới hạn khi thiếu thông tin.

Hai prompt được pull theo commit bất biến:

- V1: `nguyentatdat-2a202602578-rag-v1:d2aeb8f8ec85609d9eec40e73a96f654d37799dbc56daec4345335c179604866`.
- V2: `nguyentatdat-2a202602578-rag-v2:e770ab1a2c7f83f5258df991a1d2a08ecca6ded1b78828bb60bdb388dfbdc727`.

Đánh giá chạy cả 50 QA qua từng phiên bản, khác với routing A/B chỉ chọn một
phiên bản mỗi request. Cả hai dùng `gpt-4o-mini`, temperature 0,
`text-embedding-3-small`, FAISS, chunk 500/overlap 50/top-k 3 và cùng contexts
cho từng câu. Reference không xuất hiện trong prompt generation. RAGAS 0.4.3
dùng `gpt-4o-mini` làm evaluator; đủ bốn điểm hữu hạn cho cả 100 mẫu.

## Kết quả

| Metric | V1 | V2 | V1 − V2 |
|---|---:|---:|---:|
| Faithfulness | 0.9803 | 0.9378 | +0.0424 |
| Answer relevancy | 0.9156 | 0.8525 | +0.0631 |
| Context recall | 1.0000 | 1.0000 | 0.0000 |
| Context precision | 0.9450 | 0.9417 | +0.0033 |

Cả hai vượt faithfulness ≥0.8 và mốc thưởng ≥0.9 ở cả hai phiên bản.
V1 có faithfulness và answer relevancy trung bình cao hơn. Câu trả lời V1 dài
trung bình 315.48 ký tự; V2 là 634.26 ký tự, khoảng gấp 2.01 lần. Đây là số ký tự,
không phải số token hoặc số claims mà evaluator trích xuất.

## Giải thích chênh lệch

V1 thường trả lời trọng tâm bằng ít phát biểu hơn. V2 mở rộng định nghĩa và giải
thích, có thể thêm diễn giải không được hỗ trợ đủ rõ trong context. Đây là giả
thuyết phù hợp với kết quả và ví dụ, không phải kết luận nhân quả chỉ từ độ dài.

**Regularization:** V1 có faithfulness 1.0000 và relevancy 0.9517; V2 lần lượt
0.6000 và 0.7837. V2 thêm diễn giải L1 “driving some weights to zero”, trong khi
context nói L1 “promotes sparsity”. Phần trả lời dài có thêm phát biểu để
evaluator kiểm tra. Report không chứa nhãn entailment từng claim, nên không thể
khẳng định chính xác câu nào gây ra toàn bộ mức giảm 0.4.

**Context length:** V1 trả lời trực tiếp theo knowledge base, đạt faithfulness
1.0000 nhưng relevancy 0.7316. V2 đạt faithfulness 0.7500 và relevancy 0.9363:
phần giải thích thêm về cải thiện khả năng sinh câu trả lời mạch lạc không được
nêu rõ trong retrieved context, nhưng relevancy lại cao hơn. V1 không thắng mọi
mẫu và hai metrics đo những khía cạnh khác nhau.

Context recall bằng nhau phù hợp với thiết kế dùng chung passages và reference.
Context precision lệch khoảng 0.0033 dù retrieval giống nhau; không quy mức lệch
này thành cải thiện retriever bởi prompt. Evaluator LLM chấm riêng có thể tạo
khác biệt; một lần chạy chưa đủ đánh giá độ ổn định.

## Kết luận và giới hạn

Với bộ 50 QA này, V1 phù hợp hơn nếu ưu tiên câu trả lời ngắn, đúng trọng tâm và
grounded. V2 giảng giải rõ hơn nhưng nên giới hạn phần mở rộng ở thông tin được
context hỗ trợ. Giữ nguyên hai commit để code, report và evidence nhất quán.

Điểm do LLM chấm trên dữ liệu lab; chưa có nhiều lần chạy, khoảng tin cậy hoặc
tập kiểm tra độc lập. Faithfulness cao không chứng minh mọi câu trả lời đúng;
context recall 1.0 không chứng minh retrieval tổng quát hoàn hảo. PII và JSON
demo dùng dữ liệu giả, kiểm tra output thực tế sau `OnFailAction.FIX`.
