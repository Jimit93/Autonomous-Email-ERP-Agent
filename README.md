# Document Controller - Autonomous Email ERP Agent

**Version 5.2 (Live Monitor & WAL Enabled)**

## 📋 Description

AI-powered Email-to-ERP automation system. Processes customer emails, classifies intent, matches products, verifies inventory, generates PDF quotes/invoices, manages sales pipeline, and tracks prof[...] 

---

## 🎯 Core Capabilities

✅ **Email Processing** - IMAP ingestion, spam filtering, deduplication  
✅ **AI-Powered Intent Recognition** - LLM extraction + ML classification  
✅ **Product Matching** - Exact, fuzzy (KNN), and regex-based matching  
✅ **Inventory Management** - Stock verification, low-stock alerts, velocity calculation  
✅ **PDF Generation** - Professional quotations, POs, and invoices  
✅ **Sales Pipeline** - State machine from inquiry → invoice → ledger  
✅ **Owner Approval** - Discount negotiation, bulk order review via email  
✅ **Analytics Dashboard** - Real-time sales, inventory, and pipeline charts  
✅ **Audit Trail** - Immutable ledger, approval decisions, model feedback  

---

## 🤖 Why AI + ML = Efficiency

### **Problem Without AI/ML:**
```
if "quote" in email and "Mavic" in email:
    # Breaks with typos, context, synonyms
```
❌ Rules-based → brittle, unmaintainable, no learning  
❌ Manual categorization required  
❌ Handles <5% of real-world variations  

### **Solution With AI/ML:**

| Feature | Impact |
|---------|--------|
| **Qwen LLM** | Understands natural language context; extracts "5 units of Mavic 4 Pro" from "interested in buying 5 Mavic4 Pro" |
| **Logistic Regression** | Intent classification (~90% accuracy); catches domain-specific patterns (lpo, quotation, best price) |
| **Naive Bayes** | Spam filtering; blocks newsletters, auto-replies, phishing without false positives |
| **K-Nearest Neighbors** | Fuzzy product matching; "DJI Mavic4 Pro" → "DJI Mavic 4 Pro" (typo tolerance) |
| **Hybrid Fallback** | If LLM fails → use ML model → use regex rules → ask clarification (no single point of failure) |

### **Real-World Benefits:**

| Metric | Manual | Automated | Saving |
|--------|--------|-----------|--------|
| Quote generation | 5-10 min | 10 sec | **97%** ↓ |
| Intent classification | 2-5 min | 1 sec | **98%** ↓ |
| Inventory check | 3-5 min | Instant | **100%** ↓ |
| Email response | 10-20 min | 30 sec | **99%** ↓ |
| 24/7 operation | ❌ | ✅ | **Priceless** |

---

## 🗂️ Tech Stack

| Layer | Technology |
|-------|----------|
| **LLM** | Qwen2.5-1.5B GGUF (Hugging Face) |
| **ML** | scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, KNN) |
| **Workflow** | LangGraph state machine |
| **Email** | IMAP4/SMTP |
| **Database** | SQLite3 + WAL |
| **PDF** | WeasyPrint |
| **Dashboard** | Chart.js + HTML/CSS |
| **Async** | asyncio |


### Lightweight & Local-first

Autonomous-Email-ERP-Agent is designed to run smoothly and accurately on low-end PCs with zero cloud cost. It achieves this by combining compact, on-device ML models with FAISS-powered vector search for fast, memory-efficient retrieval. The result is a private, responsive agent that works on consumer hardware without relying on paid cloud inference.

Key optimizations:
- FAISS approximate indexing (IVF + PQ) and memory-mapped indexes
- Small/distilled models and quantized weights for fast on-device inference
- ONNX/CPU-optimized runtimes and multi-threading
- Incremental/background indexing and batched queries to preserve responsiveness

Clarification: “zero cloud cost” means no paid cloud inference or storage — local compute still consumes electricity and hardware resources.





```
Customer Email
    ↓
extract_requirements (LLM + ML intent)
    ↓
check_stock (inventory verification)
    ↓
[Stock OK?] ──Yes──> generate_quote → send (QUOTE_SENT)
     ↓
    [Bulk/Shortage?] ──Yes──> owner_approval → ask_owner_discount
                ↓
            (Owner replies: "APPROVE 10%")
                ↓
         apply_discount_and_requote
                ↓
    [Customer replies: "Approved"]
         quote_approval
                ↓
         generate_lpo (LPO_SENT)
                ↓
    [Customer: "Received"]
      delivery_confirmed
                ↓
      generate_invoice (INVOICE_SENT)
                ↓
     record_sale (Stock -qty, Ledger +revenue)
                ↓
    COMPLETED_AND_RECORDED
```

---

## 💾 Database Schema

| Table | Purpose |
|-------|--------|
| `orders` | Customer deal tracking |
| `order_items` | Line items (product, qty, discount) |
| `inventory` | Catalogue (price, stock, reorder level) |
| `sales` | Immutable ledger (revenue, profit) |
| `approval_audit` | Discount & bulk decisions |
| `backtest_runs` | ML model performance |
| `processed_messages` | Message-ID watermarks |

---

## 🧪 Quality Assurance

### Backtest Results (10 test cases):
```
Intent Accuracy: 90%
Macro F1-Score: 0.88
Bulk Detection Precision: 100%
Bulk Detection Recall: 100%
```

### Safety Features:
✅ WAL (Write-Ahead Logging) for crash-safe transactions  
✅ Foreign key constraints  
✅ Atomic stock deduction  
✅ Immutable ledger  
✅ Message-ID deduplication  
✅ No credentials in logs  

---

## 🔐 Security

- No credentials in code (uses `.env`)
- Immutable audit trail
- Transaction safety (WAL)
- Spam filtering + domain blacklist
- HTTPS-ready (SMTP STARTTLS)

---

## 📄 License

Proprietary to Aerotech Drones. All rights reserved.

---

**Built with ❤️ using Python, Qwen LLM, scikit-learn, and asyncio magic.** ✨
