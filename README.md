# Lumiere AI Service

FastAPI microservice cung cấp các tính năng AI cho hệ thống nhà hàng Lumière. Service giao tiếp nội bộ với Spring Boot backend thông qua shared service key, không expose trực tiếp ra internet.

---

## Kiến trúc tổng quan

```
Spring Boot Backend
        │  X-AI-Service-Key (internal)
        ▼
┌─────────────────────────────────────────────────────┐
│                 Lumiere AI Service                  │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Module 1    │  │  Module 2    │  │ Module 3 │  │
│  │  Foundation  │  │  Customer    │  │ Back-    │  │
│  │  & Data Sync │  │  Facing      │  │ Office   │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
│         │                 │               │         │
│    Qdrant +          RAG Search +     LightGBM /   │
│    Cohere            LLM Function     FP-Growth /  │
│    embed-v3.0        Calling          CP-SAT        │
└─────────────────────────────────────────────────────┘
        │                   │
        ▼                   ▼
    Qdrant              LLM Provider
  (vector DB)        (OpenAI-compatible)
```

Service khởi động theo cơ chế **degraded mode**: nếu Qdrant hoặc LLM không khả dụng, các module phụ thuộc sẽ trả về fallback thay vì crash toàn bộ service.

---

## Các module AI

### Module 1 — Foundation & Data Sync

**Endpoint:** `POST /ai/sync-menu`

Nhận thông tin món ăn từ backend, tạo vector embedding qua **Cohere v2 API** (model `embed-multilingual-v3.0`, 1024 chiều), và upsert vào Qdrant. Đây là bước nền tảng để các module khác hoạt động.

- Embedding model: **Cohere `embed-multilingual-v3.0`** (gọi qua Cohere v2 API)
- Vector DB: Qdrant (collection `menu_items`, HNSW m=16/ef_construct=200, Cosine)
- Full-text được embed: `"Tên: {name}, Giá: {price}, Mô tả: {desc}, Danh mục: {cat}, Đặc tính: {tags}"`
- Metadata lưu kèm: `menu_item_id`, tên, giá, mô tả, category, tags, full_text
- **Auto-seed:** Service tự động kéo toàn bộ menu từ backend khi khởi động nếu Vector DB đang rỗng

### Module 2 — Customer Facing

**`POST /ai/recommend`** — Gợi ý món ăn

1. Lấy vector 1024-dim của từng món trong giỏ hàng từ Qdrant
2. Tính centroid = element-wise mean của tất cả vectors
3. Cosine similarity search với centroid (fetch top_k×3, exclude cart items)
4. Đọc CTR từ Redis (`ai:ctr:clk / ai:ctr:imp`)
5. Blended score = **0.6 × vector_score + 0.4 × CTR**
6. Same-category penalty ×0.5 (ưu tiên cross-sell); `reason = "combo+popularity"` hoặc `"similar"`

**`POST /ai/chatbot`** — Chatbot đặt món

Pipeline RAG ba bước:
1. **Embed query** qua Cohere API (`input_type="search_query"`)
2. **Hybrid Search RRF** (K=60): dense Qdrant search + BM25 keyword search trên toàn bộ documents → fused top 5
3. **LLM Function Calling** với tool `add_to_cart(menu_item_id, quantity)` — timeout 15s. Tool calls → `ADD_ITEM:{id}:{qty}` actions. Hỗ trợ tiếng Việt.

### Module 3 — Back Office

**`POST /ai/forecast`** — Dự báo đơn hàng / doanh thu

1. Kéo lịch sử từ backend (min max(90, horizon+30) ngày)
2. Feature engineering: lag_1..lag_7, roll_mean_7, roll_std_7, day_of_week, is_weekend, month (12 features)
3. Ưu tiên dùng **pre-trained model** từ disk (`models/forecast_{metric}.joblib`); nếu chưa có → train LightGBM on-demand (n_estimators=150, lr=0.05, num_leaves=15)
4. **Recursive prediction**: mỗi predicted value append vào window dùng làm lag tiếp theo
5. Confidence interval: ±1.5 × residual_std. Cần ≥ 14 ngày dữ liệu, fallback = đường phẳng.

**`POST /ai/combo-generate`** — Tạo gợi ý combo

1. Ưu tiên dùng **pre-trained rules** từ disk (`models/combo_rules.pkl`) nếu có
2. Nếu không: kéo order-items, build transaction sets (order_id → frozenset{menu_item_id})
3. One-hot encode → FP-Growth (min_support threshold) → Association Rules (min_confidence)
4. Lọc lift > 1.0, deduplicate bằng frozenset, tối đa 20 combos

**`POST /ai/kitchen-batching`** — Tối ưu ghép lệnh bếp

1. Nhóm active tasks theo menu_item_id (chỉ nhóm ≥ 2 tasks)
2. Tính score mỗi nhóm: `α(2)×T_wait + β(5)×N_tasks − γ(15)×P_penalty` (T_wait cap 60 phút, P_penalty khi vượt capacity=5)
3. **CP-SAT solver**: maximize tổng score, constraint Σ x_i ≤ 5, timeout 0.5s
4. Saving estimate: `min(15.0, (N−1)×avg_cook_time/60)` phút

---

## Cấu trúc thư mục

```
lumiere-ai-service/
├── app/
│   ├── main.py                 # App factory, lifespan (startup/shutdown + auto-seed)
│   ├── settings.py             # Pydantic settings đọc từ .env
│   ├── errors.py               # AiErrorBody schema chuẩn hoá lỗi
│   ├── api/
│   │   └── ai.py               # Toàn bộ route definitions
│   ├── schemas/
│   │   ├── ai.py               # Request / Response Pydantic models
│   │   └── common.py
│   ├── services/
│   │   ├── embedder.py         # sentence-transformers wrapper (thread-safe)
│   │   ├── vector_store.py     # Qdrant client wrapper (HNSW, hybrid search)
│   │   ├── sync.py             # sync-menu logic + full-sync từ backend
│   │   ├── recommend.py        # Centroid vector similarity
│   │   ├── chatbot.py          # RAG + LLM function calling
│   │   ├── forecast.py         # LightGBM time-series forecasting
│   │   ├── combo.py            # FP-Growth association rule mining
│   │   ├── batching.py         # CP-SAT kitchen batching optimiser
│   │   ├── feedback.py         # CTR-based re-ranking (Phase 2)
│   │   ├── ctr_store.py        # Click-Through Rate accumulator
│   │   ├── retrain.py          # Batch retraining orchestration (Phase 3)
│   │   ├── job_store.py        # Async job status (PENDING/PROCESSING/COMPLETED/FAILED)
│   │   └── redis_client.py     # Async Redis client (CTR + job store)
│   ├── clients/
│   │   ├── backend.py          # HTTP client gọi Spring Boot internal APIs
│   │   └── llm.py              # OpenAI-compatible LLM client
│   ├── middleware/
│   │   ├── rate_limit.py       # Per-IP rate limiting
│   │   └── request_logging.py  # Structured request/response logging
│   ├── security/
│   │   └── service_key.py      # X-AI-Service-Key header guard
│   └── utils/
│       └── trace.py            # Request trace ID
├── tests/
│   └── test_contract.py        # Contract tests (auth, schema, happy paths)
├── pyproject.toml
├── .env.example
├── docker-compose.yml          # Qdrant local setup
└── README.md
```

---

## Cài đặt & Chạy

### Yêu cầu

- Python 3.12+
- Qdrant (local hoặc cloud)
- Redis (local hoặc cloud) — cần cho feedback và retrain
- LLM provider hỗ trợ OpenAI-compatible API (cho chatbot)
- Spring Boot backend đang chạy (cần cho auto-seed)

### Bước 1 — Cài dependencies

```bash
cd lumiere-ai-service
pip install -e ".[dev]"
```

### Bước 2 — Cấu hình môi trường

Sao chép `.env.example` thành `.env` và điền các giá trị:

```env
# Shared secret với Spring Boot backend (bắt buộc)
AI_SERVICE_KEY=your-secret-key

# Server
HOST=0.0.0.0
PORT=8001

# Spring Boot backend (bắt buộc để auto-seed và export data)
BACKEND_BASE_URL=http://localhost:8080/api/v1

# LLM provider (cho chatbot) — phải hỗ trợ OpenAI-compatible API + function calling
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.your-llm-provider.com/v1
LLM_API_KEY=your-llm-api-key
LLM_MODEL=your-model-name

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                    # để trống nếu dùng local
QDRANT_COLLECTION=menu_items

# Embedding — Cohere v2 API (thay đổi EMBED_MODEL cần xoá và tạo lại Qdrant collection)
COHERE_API_KEY=your-cohere-api-key
EMBED_MODEL=embed-multilingual-v3.0

# Redis (cho CTR feedback + retrain job store)
REDIS_URL=redis://localhost:6379

# Thư mục lưu trained models (forecast_orders.joblib, forecast_revenue.joblib, combo_rules.pkl)
MODEL_DIR=models
```

### Bước 3 — Khởi động Qdrant (local)

```bash
docker-compose up -d
```

Kiểm tra Qdrant đã sẵn sàng:

```bash
curl http://localhost:6333/healthz
```

### Bước 4 — Khởi động AI Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Lần đầu khởi động** sẽ thực hiện theo thứ tự:

1. **Kiểm tra Cohere API**: gọi `embed_one("ping")` để xác nhận COHERE_API_KEY hợp lệ và auto-detect vector dimension (1024 chiều với `embed-multilingual-v3.0`)
2. **Kết nối Qdrant**: tạo collection `menu_items` nếu chưa có (HNSW m=16, ef_construct=200, Cosine)
3. **Kết nối Redis**: dùng cho CTR store và job status
4. **Auto-seed**: Nếu Qdrant collection rỗng → `asyncio.create_task(sync_all_menu_items_from_backend())` — batch embed tất cả menu items bằng Cohere API, upsert vào Qdrant. Service không chờ sync hoàn tất.

Log startup thành công sẽ trông như sau:

```
INFO  Embedder ready: model=embed-multilingual-v3.0  dim=1024
INFO  VectorStore ready: collection=menu_items  dim=1024
INFO  Vector store is empty — triggering auto-seed from backend
INFO  Service started: embedder=UP  vector_store=UP  redis=UP
INFO  Synced 84 menu items from backend       ← log xuất hiện sau vài giây
```

Nếu Cohere API key không hợp lệ hoặc backend chưa chạy: service vẫn start nhưng `sync-menu` và `recommend` không khả dụng (log WARNING). Khi backend sẵn sàng, gọi thủ công `POST /ai/sync-menu/full`.

**Swagger UI:** `http://localhost:8001/docs`

### Bước 5 — Kiểm tra trạng thái

```bash
curl -H "X-AI-Service-Key: your-secret-key" http://localhost:8001/ai/health
```

Response mẫu khi service hoạt động đầy đủ:

```json
{
  "status": "UP",
  "version": "v2",
  "components": { "embedder": true, "vector_store": true, "llm": true },
  "endpoints": {
    "sync_menu": true, "recommend": true, "chatbot": true,
    "forecast": true, "combo_generate": true, "kitchen_batching": true
  }
}
```

### Bước 6 — Đồng bộ dữ liệu thủ công (nếu cần)

Nếu auto-seed chưa chạy hoặc cần re-sync toàn bộ menu sau khi có thay đổi lớn:

```bash
curl -X POST \
  -H "X-AI-Service-Key: your-secret-key" \
  http://localhost:8001/ai/sync-menu/full
```

Response: `{"success": true, "message": "Full sync started in background"}`

Quá trình sync chạy nền — kiểm tra log để theo dõi tiến trình.

---

## Bảo mật

Tất cả endpoint đều yêu cầu header:

```
X-AI-Service-Key: <AI_SERVICE_KEY>
```

Thiếu hoặc sai key sẽ nhận `401 / 403`. Key này được chia sẻ giữa AI service và Spring Boot backend, không expose ra client.

---

## API Reference

### Health

```
GET /ai/health
```

Trả về trạng thái từng component (embedder, vector_store, llm) và danh sách endpoint nào đang hoạt động.

---

### POST /ai/sync-menu

Upsert một món ăn vào vector store. Backend gọi endpoint này mỗi khi tạo / cập nhật món (event-driven).

```json
// Request
{
  "menu_item_id": 42,
  "name": "Risotto nấm truffle",
  "description": "Risotto kem với nấm truffle đen",
  "price": 285000,
  "category": "Main Course",
  "tags": ["vegetarian", "signature"]
}

// Response
{ "success": true, "menu_item_id": 42, "vector_id": "vec_42" }
```

---

### DELETE /ai/sync-menu/{item_id}

Xoá một món ăn khỏi vector store. Backend gọi khi xoá món.

```
DELETE /ai/sync-menu/42
```

```json
{ "success": true, "menu_item_id": 42, "vector_id": "vec_42" }
```

---

### POST /ai/sync-menu/full

Kéo toàn bộ menu từ backend và upsert vào vector store. Chạy nền, trả về ngay.

```json
{ "success": true, "message": "Full sync started in background" }
```

Dùng khi: lần đầu khởi động (nếu auto-seed bị bỏ qua), sau migration dữ liệu lớn, hoặc khi cần reset lại toàn bộ vector store.

---

### POST /ai/recommend

Gợi ý món dựa trên giỏ hàng hiện tại.

```json
// Request
{ "current_items": [1, 5, 12], "top_k": 3 }

// Response
{
  "success": true,
  "source": "model",
  "model_version": "recommend_centroid_v1",
  "items": [
    { "menu_item_id": 8, "score": 0.872, "reason": "combo+popularity" },
    { "menu_item_id": 23, "score": 0.801, "reason": "combo+popularity" }
  ]
}
```

---

### POST /ai/chatbot

Chatbot đặt món tiếng Việt. `suggested_actions` chứa lệnh `ADD_ITEM:<id>:<qty>` để frontend xử lý.

```json
// Request
{
  "session_id": "sess_abc123",
  "message": "cho mình 2 risotto nấm truffle",
  "current_cart_item_ids": [5]
}

// Response
{
  "success": true,
  "reply_text": "Dạ em đã thêm 2 Risotto nấm truffle vào giỏ hàng cho bạn ạ!",
  "suggested_actions": ["ADD_ITEM:42:2"]
}
```

---

### POST /ai/forecast

Dự báo đơn hàng hoặc doanh thu theo ngày.

```json
// Request
{ "metric": "orders", "horizon_days": 7 }

// Response
{
  "success": true,
  "metric": "orders",
  "predictions": [
    { "day": 1, "value": 84.5, "lower_bound": 67.2, "upper_bound": 101.8 },
    { "day": 2, "value": 91.0, "lower_bound": 73.7, "upper_bound": 108.3 }
  ]
}
```

`metric`: `"orders"` hoặc `"revenue"` — `horizon_days`: 1–365

---

### POST /ai/combo-generate

Phân tích order history, tìm combo món hay được gọi cùng nhau.

```json
// Request
{ "analyze_days": 30, "min_support": 0.05, "min_confidence": 0.6 }

// Response
{
  "success": true,
  "draft_combos": [
    { "combo_items": [3, 17], "confidence_score": 0.78, "lift_score": 2.34 },
    { "combo_items": [3, 17, 42], "confidence_score": 0.65, "lift_score": 1.91 }
  ]
}
```

---

### POST /ai/kitchen-batching

Gợi ý ghép các task bếp cùng món để nấu một lần, giảm thời gian chờ.

```json
// Request
{
  "active_tasks": [
    { "task_id": 501, "menu_item_id": 7, "quantity": 1, "created_at": "2026-01-01T08:20:00Z", "cook_time_seconds": 900 },
    { "task_id": 502, "menu_item_id": 7, "quantity": 1, "created_at": "2026-01-01T08:22:00Z", "cook_time_seconds": 900 }
  ]
}

// Response
{
  "success": true,
  "suggestions": [
    {
      "menu_item_id": 7,
      "task_ids": [501, 502],
      "estimated_saving_minutes": 15.0,
      "reason": "Same item high queue overlap"
    }
  ]
}
```

---

### POST /ai/feedback

Nhận feedback CTR (Click-Through Rate) từ backend để re-rank gợi ý (Phase 2).

```json
// Request
{
  "events": [
    { "type": "RECOMMENDATION_CLICK", "shown": [1, 2, 3], "clicked": [2] }
  ]
}

// Response
{ "success": true, "processed": 1 }
```

---

### POST /ai/retrain

Kích hoạt re-train model combo (FP-Growth) và forecast (LightGBM) từ order history mới nhất. Chạy nền, trả về `job_id` để theo dõi (Phase 3).

```json
// Response
{ "success": true, "job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "PENDING" }
```

---

### GET /ai/jobs/{job_id}

Kiểm tra trạng thái job retrain.

```json
// Response
{ "success": true, "job_id": "550e8400...", "status": "COMPLETED", "message": "Synced 84 items, retrained combo+forecast" }
```

`status`: `PENDING` → `PROCESSING` → `COMPLETED` / `FAILED`

---

## Chạy Tests

```bash
pytest tests/test_contract.py -v
```

Tests dùng `TestClient` (không cần server thật), tự inject service key. Bao phủ: auth guard, health, schema validation, và happy path của tất cả endpoint.

---

## Dependency chính

| Thư viện | Mục đích |
|---|---|
| FastAPI + Uvicorn | Web framework & ASGI server |
| httpx | Async HTTP client — gọi Cohere API, LLM provider, backend export |
| qdrant-client ≥ 1.11 | Async Qdrant client (HNSW vector search) |
| rank-bm25 | BM25Okapi keyword search (hybrid search trong chatbot) |
| lightgbm | LGBMRegressor cho time-series forecasting |
| joblib | Save/load LightGBM model files (.joblib) |
| mlxtend | FP-Growth + association_rules cho combo discovery |
| ortools | CP-SAT solver cho kitchen batching |
| redis[asyncio] | Async Redis — CTR counters (ai:ctr:*) + job status (ai:job:*) |
| pandas / numpy | Feature engineering (forecast) và one-hot encoding (combo) |
| pydantic-settings | Type-safe .env config |
