# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603452
- **Họ và tên:** Nguyễn Tiến Đạt
- **SEED** (in ở ô đầu notebook): 9154

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output (15/15 bộ kiểm tra đạt) | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc1.jpg`](hand_calc1.jpg), [`hand_calc2.jpg`](hand_calc2.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán (số liệu từ ô "Bảng so sánh" của notebook, SEED=9154):

| Cách làm | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 1.000 | 0.867 | 0.929 | 0.947 | 10.5 | 1,879.9 | 6.07 | 0 |
| 2. BoW + MLP (deep learning) | 0.978 | 0.908 | 0.942 | 1.000 | 22.9 | 28,973.1 | 0.38 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.903 | 0.949 | 0.925 | 1.000 | 125.7 | 24.0 | 60.77 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.968 | 0.939 | 0.953 | 0.974 | 126.1 | 27.2 | 56.66 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.989 | 0.918 | 0.952 | 0.974 | 127.0 | 26.7 | 62.55 | 0 |
| 4a. LLM nhỏ zero-shot | 0.952 | 1.000 | 0.976* | 0.976 | 0 | 1.98 | 846.61 | 0** |
| 4b. LLM lớn zero-shot | 0.909 | 1.000 | 0.952* | 0.952 | 0 | 2.28 | 657.06 | 0** |
| 4c. LLM nhỏ few-shot | 0.909 | 1.000 | 0.952* | 0.952 | 0 | 0.99 | 2389.69 | 0** |

\* Hàng LLM chỉ chạy trên 40 tin SUB nên cột "F1 (test)" thực chất là F1 trên SUB.
\** Chạy trên Groq free tier nên chi phí hiển thị 0; nếu tính theo bảng giá thì chi phí thực tế nhỏ hơn 0,01 USD cho 40 tin.

Nhận xét nhanh:
- **Chất lượng:** 4a (LLM nhỏ zero-shot) dẫn đầu trên SUB (F1 0.976), nhưng chỉ hơn 3b (0.974) đúng 0.002 — nằm trong nhiễu của 40 tin. Trên toàn bộ test (775 tin), 3b dẫn đầu với F1 0.953.
- **Tốc độ:** chỉ BoW + LogReg và BoW + MLP đạt SLA p95 < 50 ms/tin; toàn bộ nhóm embedding (p95 56–63 ms) và LLM (657–2390 ms) đều vượt.
- **Chi phí:** 4 cách đầu gần như miễn phí. Riêng LLM, nếu tự trả tiền API thì chi phí tỉ lệ với số tin (xem mục 6).
- **Đánh đổi:** không có cách nào vừa nhanh nhất vừa chính xác nhất. BoW + MLP là điểm cân bằng tốt nhất trong miền đạt SLA.

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất:

1. **`llm_models` (Thí nghiệm 4.A)** — Dự đoán model lớn `gpt-oss-120b` sẽ có F1 cao hơn model nhỏ `gpt-oss-20b` khoảng 0.02–0.05. Thực tế **ngược lại**: nhỏ = 0.976, lớn = 0.952. Model lớn bỏ sót thêm 2 tin ham ngắn, viết tắt/teencode ("Die... Now i have e toot fringe again..."), tức là trên 40 tin thì chênh 1–2 tin đã đảo thứ hạng. Bài học: với tập đánh giá nhỏ, "model to hơn" không đảm bảo tốt hơn.

2. **`hard_eval` (Stress test 50 case)** — Dự đoán BoW 55–65%, embedding 68–75%, LLM 65–80%. Thực tế BoW tụt sâu hơn dự đoán (LogReg 50%, MLP 58%) còn LLM **cao hơn hẳn** dự đoán (zero-shot 90%, có prompt phòng thủ 92%). Tôi đã đánh giá thấp khả năng "đọc như người" của LLM và đánh giá thấp mức độ phụ thuộc từ khoá của BoW.

3. **`tradeoff` (đường Pareto)** — Dự đoán Pareto gồm BoW + LogReg, BoW + MLP và 3b, với 3b ở vị trí cân bằng. Thực tế **chỉ BoW + MLP nằm trên Pareto**: BoW + LogReg bị MLP lấn át (chậm hơn khi gọi từng tin: p95 6.07 ms so với 0.38 ms, F1 lại thấp hơn), còn 3b/3c/kNN đều vượt SLA nên không cạnh tranh được trong miền này. Nguyên nhân: tôi dự đoán theo "cảm giác" về F1 mà quên rằng p95 của encoder MiniLM trên CPU đã ~57 ms.

Ngoài ra `dl_curve` cũng lệch nhẹ: tôi đoán overfit sau epoch 15–20, nhưng val loss thấp nhất ở **epoch 9** — mô hình bắt đầu overfit sớm hơn dự đoán.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:** BoW + LogReg — F1 từ 0.947 (SUB) xuống **0.286** trên 50 case khó, accuracy chỉ 50%. BoW + MLP cũng tụt mạnh: 1.000 → 0.432. Nhóm embedding đỡ hơn (F1 50 case: 3a kNN 0.764; 3b 0.640; 3c 0.667), còn LLM gần như giữ nguyên chất lượng (F1 4a 0.918; 4d 0.935 — accuracy 90%/92%).

Lý do điểm test thường bị "quá lạc quan": tập test được chia ngẫu nhiên cùng phân phối với train, spam trong đó vẫn đầy từ khoá mạnh (`free`, `win`, `prize`, `call`) và phần lớn là mẫu dễ. BoW chỉ cần khớp đúng token là thắng, nên điểm cao không chứng minh mô hình hiểu nghĩa. 50 case khó được viết tay để phá đúng các shortcut đó (spam giọng tự nhiên, né từ khoá, prompt injection, tin ham trông giống spam).

**Hai case cụ thể:**

- **Case #2 — "FREE pizza in the office kitchen right now, come quick before it's gone" (ham).** 4d (LLM + prompt phòng thủ) vẫn đoán spam. Cơ chế: prompt phòng thủ dạy model coi từ ngữ khẩn cấp/mời chào là dấu hiệu spam; ở đây "FREE" + "right now" + "before it's gone" trùng khớp đúng mẫu đó, trong khi tín hiệu ngữ cảnh (office kitchen, đồng nghiệp) yếu hơn. Đây là cái giá của việc phòng thủ quá tay: tăng đúng nhóm injection nhưng làm hỏng một phần nhóm "tưởng spam nhưng không phải".

- **Case #47 — "lol I just got a text saying I won £1000 and to call some 0906 number..." (ham).** Cả BoW và LLM đều dễ sai vì câu chứa gần như trọn bộ từ khoá spam (`won`, `£1000`, `call`, `0906`). BoW không có khái niệm "trích dẫn"; LLM có ngữ cảnh nhưng vẫn bị nội dung spam lấn át. Đúng như nhận xét ở câu hỏi 6: nhóm "Nói về spam" là điểm yếu chung của mọi mô hình (embedding 50%, 4d 75%).

**Theo nhóm case (từ bảng % đoán đúng của notebook):**
- BoW yếu nhất ở *Spam giọng tự nhiên* (LogReg 0%, MLP 10%) và *Né từ khoá* (LogReg 17%, MLP 0%) — vì không có token spam nào khớp, hoặc token bị vỡ do leetspeak/chèn ký tự.
- Embedding yếu nhất ở *Tưởng spam nhưng không phải* (70–80%) và *Nói về spam* (50%) — vector nghĩa kéo tin ham có `free`/`prize` về gần vùng spam.
- LLM mạnh nhất nhưng vẫn sai ở *Prompt injection* khi zero-shot (83%); prompt phòng thủ đưa nhóm này lên 100%, đổi lại giảm nhẹ *Spam giọng tự nhiên* (100% → 90%) và *Nói về spam* (100% → 75%).

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 tin/s trung bình), p95 < 50 ms/tin.

**Dựa trên số liệu đo được:**
- **Đạt SLA độ trễ:** chỉ BoW + LogReg (p95 6.07 ms, 1,880 tin/s) và BoW + MLP (p95 0.38 ms, 28,973 tin/s). Nhóm embedding p95 56–63 ms và LLM 657–2,390 ms đều **không đạt**.
- **Đạt cả độ trễ lẫn chất lượng (trong miền đạt SLA):** BoW + MLP — F1 test 0.942, accuracy 98.6%, precision 0.978, recall 0.908, miễn phí, không phụ thuộc mạng.
- **Chất lượng cao nhất nhưng không đạt SLA:** 4a LLM nhỏ zero-shot (F1 SUB 0.976) — chậm gấp ~2,200 lần BoW + MLP khi gọi từng tin.

**Chi phí nếu dùng LLM:** với `gpt-6-luna` ($0.10/$0.50 mỗi 1M token), mỗi tin 100 token input + 1 token output → $0.0000105/tin → 1 triệu tin/ngày = $10.50/ngày → **10 triệu tin/ngày ≈ $105/ngày ≈ $3,150/tháng** (chưa kể retry và traffic thực tế). Ngoài ra cần ~58 request song song liên tục để đạt 116 tin/s, và mọi sự cố mạng/API đều làm nghẽn hệ thống.

**Kết luận:** chọn **BoW + MLP làm lớp lọc chính** vì đạt SLA, miễn phí và ổn định. Không dùng LLM làm lớp kiểm tra duy nhất; chỉ dùng có kiểm soát trong kiến trúc lai dưới đây.

**Thiết kế lai (router) đề xuất:**
1. **Lớp 1 — BoW + MLP (hoặc LogReg):** chạy trên 100% lưu lượng, ~0.4 ms/tin. Nếu `p(spam) < 0.3` → ham chắc chắn, cho qua; nếu `p > 0.7` → spam chắc chắn, chặn luôn.
2. **Lớp 2 — LLM chỉ cho vùng lưỡng lự `p ∈ [0.3, 0.7]`** (ước tính 5–10% lưu lượng = 0.5–1 triệu tin/ngày ≈ 6–12 tin/s). Với ~10 request song song ở p95 ~0.85 s, LLM xử lý được ~12 tin/s — vừa đủ. Chi phí giảm còn ~$5–10/ngày (~$150–300/tháng).
3. **Lớp 3 — quy tắc/human review** cho tin bị chặn có rủi ro cao (OTP, ngân hàng, giao dịch) và cho các case LLM trả lời không hợp lệ.

Router tốt hơn dùng một cách duy nhất vì: BoW + MLP lo được đa số tin với chi phí gần bằng 0 và độ trễ không đáng kể; LLM chỉ dùng cho phần khó — nơi nó thực sự tạo giá trị (F1 50 case khó 0.918–0.935 so với 0.286–0.432 của BoW). Như vậy vừa giữ SLA, vừa giảm chi phí, vừa tăng chất lượng ở đúng vùng mơ hồ.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1–3 (vùng tự code) | "Giải thích sigmoid, gradient descent của logistic regression và vì sao dùng BCE" | Giải thích công thức `σ(z)`, đạo hàm `Xᵀ(p−y)/n`, ý nghĩa precision/recall/F1 | Tự viết `vectorize`, `sigmoid`, `train_logreg`, `metrics`, `cosine_similarity`, `top_k_similar`, `knn_predict`; đối chiếu bằng tay với test của notebook (15/15 đạt) |
| Phần 2 | "Vì sao val loss ngừng giảm dù train loss vẫn giảm?" | Giải thích overfitting, gợi ý early stopping | Kiểm tra trên biểu đồ: val loss thấp nhất epoch 9; tự chọn epoch dừng và ghi vào câu hỏi 2.A |
| Phần 4 (được dùng AI) | "Viết `call_openai`, prompt zero/few-shot và `parse_label` theo template LLM API" | Sinh khung hàm `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Tự review: sửa `parse_label` để xử lý "not spam" trước "spam", thêm word boundary; kiểm tra bằng bộ test giả lập của notebook |
| Phần 5 | "Gợi ý cách phân tích 50 case khó theo nhóm lỗi" | Gợi ý nhóm theo cơ chế (natural_spam, obfuscation, injection…) và cách đọc bảng % | Tự tính lại số liệu từ output, tự viết nhận xét và 7 case khó của riêng mình (`MY_CASES`) |

**Điều đã học được hôm nay (mà trước đó chưa hiểu):**
1. **Điểm test cao chưa chứng minh mô hình hiểu ngôn ngữ.** BoW + MLP đạt F1 1.000 trên SUB nhưng chỉ 0.432 trên 50 case khó — chất lượng thật nằm ở phân phối dữ liệu, không phải con số đẹp.
2. **"Thêm một lớp / model to hơn" không tự động tốt hơn.** Bỏ ReLU cho F1 y hệt 0.942; LLM lớn lại kém LLM nhỏ trên 40 tin. Điều quyết định là dữ liệu, cách chia tập và độ phù hợp của kiến trúc với bài toán.
3. **Đánh đổi kỹ thuật phải đo bằng số.** Trước đây tôi nghĩ embedding là lựa chọn "vừa nhanh vừa tốt"; thực tế p95 ~57 ms khiến nó vượt SLA 50 ms, còn LLM tốt nhất lại chậm gấp hàng nghìn lần. Vì vậy bài toán sản phẩm cần router: mô hình rẻ xử lý phần lớn, mô hình đắt chỉ xử lý vùng mơ hồ.
