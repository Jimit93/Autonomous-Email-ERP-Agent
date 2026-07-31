"""
=============================================================================
AEROTECH DRONES - AUTONOMOUS EMAIL ERP AGENT
Version 5.2 (Live Monitor & WAL Enabled)
=============================================================================
"""

import nest_asyncio
nest_asyncio.apply()

import os
import re
import json
import uuid
import html
import email
import email.utils
import email.header
import asyncio
import imaplib
import smtplib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Dict, Any, Literal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from weasyprint import HTML
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import MultinomialNB

# --- live monitor hooks (all fail-safe; see agent_trace.py) ---
from agent_trace import (ensure_trace_schema, record_trace, heartbeat,
                         export_models)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

MAILBOX = os.getenv("MAILBOX", "INBOX")

if not EMAIL_USER or not EMAIL_PASS:
    raise ValueError("CRITICAL: EMAIL_USER or EMAIL_PASS missing from .env file!")

DB_FILE = os.getenv("DB_FILE", "aerotech_v5.db")

CURRENCY = "\u20b9"
TAX_LABEL = "GST"
TAX_RATE = 0.18

KNN_MAX_DISTANCE = 0.35
MAX_EMAILS_PER_CYCLE = 10
MAX_BODY_CHARS_FOR_LLM = 1200
POLL_SECONDS = 10                 
INVENTORY_SWEEP_SECONDS = 60

DEFAULT_REORDER_LEVEL = 10        
VELOCITY_WINDOW_DAYS = 30         
LOW_STOCK_URGENT_DAYS = 14        

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")

IGNORE_DOMAINS = [
    "temuemail.com", "wio.io", "newsletter", "no-reply", "noreply",
    "promotions", "marketing", "notifications", "support@github.com",
    "google.com", "facebookmail.com", "linkedin.com", "twitter.com",
    "instagram.com",
]


# =============================================================================
# 2. MACHINE LEARNING SETUP
# =============================================================================
# Catalogues are loaded from SQL after database initialization.
inventory_items = []
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = None
nn = None


training_emails = [
    "Please send a quote for DJI Mavic 4 Pro", "I want to order 2 HoverAir drones",
    "Approved, please send the invoice", "Can you give me a 10% discount on the Neo 2?",
    "Do you have the Matrice 4T in stock?", "Send the local purchase order for the drone",
    "You have won a free iPhone click here", "Weekly newsletter top 10 tech trends",
    "Let's schedule a meeting for SEO optimization services", "Limited time offer on car insurance",
    "Hi, I would like to offer web development services for your company",
    "Unsubscribe from this mailing list",
    "Your package could not be delivered, please pay customs fee"
]
email_labels = ["valid"] * 6 + ["spam"] * 7
spam_vectorizer = TfidfVectorizer(stop_words="english")
spam_classifier = MultinomialNB()
spam_classifier.fit(spam_vectorizer.fit_transform(training_emails), email_labels)

COMMERCIAL_SIGNALS = [
    "quote", "quotation", "price", "pricing", "cost", "order", "purchase",
    "buy", "invoice", "lpo", "stock", "drone", "discount", "delivery",
    "approve", "approved", "confirm"
]


# =============================================================================
# 3. TEXT HELPERS
# =============================================================================
NEGATION_RE = re.compile(
    r"\b(not|no|never|dont|don't|doesn't|didn't|cannot|can't|won't|wouldn't|"
    r"hold|stop|without|unless|before)\b", re.IGNORECASE)


def has_words(text: str, words: List[str], respect_negation: bool = True) -> bool:
    if not text:
        return False
    for w in words:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
            if respect_negation:
                window = text[max(0, m.start() - 45):m.start()]
                if NEGATION_RE.search(window):
                    continue
            return True
    return False


def clean_body(body: str, limit: int = MAX_BODY_CHARS_FOR_LLM) -> str:
    out = []
    for line in (body or "").splitlines():
        s = line.strip()
        if s.startswith(">"):
            break
        if re.match(r"^On .{5,120}\s+wrote:\s*$", s):
            break
        if re.match(r"^-{2,}\s*(Original Message|Forwarded message)", s, re.I):
            break
        if s.lower().startswith("from:") and out:
            break
        out.append(line)
    return "\n".join(out).strip()[:limit]


def decode_header_value(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        parts = email.header.decode_header(raw)
        return "".join(
            p.decode(enc or "utf-8", errors="ignore") if isinstance(p, bytes) else p
            for p, enc in parts).strip()
    except Exception:
        return str(raw)


def parse_discount(body: str) -> float:
    head = "\n".join(clean_body(body).splitlines()[:3])
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", head) or \
        re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", head)
    return min(max(float(m.group(1)), 0.0), 100.0) if m else 0.0


def scan_products_in_text(text: str) -> List[Dict[str, Any]]:
    """Match requested products against the live SQL catalogue, never a hardcoded list."""
    found = []
    catalog = get_catalog()
    for inv in catalog:
        core = re.sub(r"\(.*?\)", "", inv["product_name"]).strip()
        if not re.search(rf"\b{re.escape(core)}\b", text or "", re.IGNORECASE):
            continue
        qty = 1
        m = re.search(rf"(\d{{1,4}})\s*(?:x|units?|pcs?|nos?|pieces?)?\s+{re.escape(core)}", text, re.IGNORECASE)
        if m:
            qty = max(1, int(m.group(1)))
        found.append({"product": inv["product_name"], "quantity": qty,
                      "price": inv["selling_price"], "stock": inv["total_stock"]})
    return found


def money(v: float) -> str:
    return f"{CURRENCY}{v:,.2f}"


# =============================================================================
# 4. DATABASE
# =============================================================================
@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database():
    with db_conn() as conn:
        c = conn.cursor()
        
        # EXPLICIT WAL ENABLING 
        c.execute("PRAGMA journal_mode=WAL")
        
        c.execute("""CREATE TABLE IF NOT EXISTS orders (
                        email TEXT PRIMARY KEY, client_name TEXT, status TEXT,
                        last_updated TEXT, quotation_sent TEXT, lpo_sent TEXT,
                        invoice_sent TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS order_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, order_email TEXT,
                        product_name TEXT, quantity INTEGER, price_each REAL,
                        discount_percent REAL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS inventory (
                        product_name TEXT PRIMARY KEY, buying_price REAL,
                        selling_price REAL, total_stock INTEGER, total_sales INTEGER,
                        reorder_level INTEGER DEFAULT 10,
                        low_stock_alerted INTEGER DEFAULT 0,
                        active INTEGER NOT NULL DEFAULT 1)""")
        c.execute("""CREATE TABLE IF NOT EXISTS bot_settings (
                        key TEXT PRIMARY KEY, value TEXT)""")

        c.execute("""CREATE TABLE IF NOT EXISTS sales (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_email TEXT NOT NULL, client_name TEXT,
                        product_name TEXT NOT NULL, quantity INTEGER NOT NULL,
                        unit_price REAL NOT NULL,
                        discount_percent REAL NOT NULL DEFAULT 0,
                        buying_price REAL NOT NULL DEFAULT 0,
                        net_revenue REAL NOT NULL, gross_profit REAL NOT NULL,
                        invoice_ref TEXT, sold_at TEXT NOT NULL,
                        stock_applied INTEGER NOT NULL DEFAULT 0)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sold_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sales_pending ON sales(stock_applied)")

        c.execute("""CREATE TABLE IF NOT EXISTS processed_messages (
                        message_id TEXT PRIMARY KEY, seen_at TEXT)""")

        existing_cols = [r[1] for r in c.execute("PRAGMA table_info(inventory)").fetchall()]
        if "reorder_level" not in existing_cols:
            c.execute(f"ALTER TABLE inventory ADD COLUMN reorder_level INTEGER "
                      f"DEFAULT {DEFAULT_REORDER_LEVEL}")
            c.execute("UPDATE inventory SET reorder_level = "
                      "MAX(1, MIN(?, CAST(total_stock * 0.25 AS INTEGER)))",
                      (DEFAULT_REORDER_LEVEL,))
            logging.info("Migration: inventory.reorder_level added and back-filled.")
        if "low_stock_alerted" not in existing_cols:
            c.execute("ALTER TABLE inventory ADD COLUMN low_stock_alerted INTEGER DEFAULT 0")
            logging.info("Migration: inventory.low_stock_alerted added.")
        if "active" not in existing_cols:
            c.execute("ALTER TABLE inventory ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
            logging.info("Migration: inventory.active added.")

        # Do not seed a hardcoded product list. Inventory is populated and maintained
        # through the SQL-backed owner management helpers below.



def get_catalog(include_inactive: bool = False) -> List[dict]:
    """Return the live product catalogue and stock from SQL."""
    with db_conn() as conn:
        query = "SELECT product_name, buying_price, selling_price, total_stock, total_sales, reorder_level, low_stock_alerted FROM inventory"
        if not include_inactive:
            query += " WHERE COALESCE(active, 1)=1"
        query += " ORDER BY product_name"
        return [dict(row) for row in conn.execute(query).fetchall()]


def refresh_catalog_models() -> None:
    """Rebuild product matching models from current SQL rows."""
    global inventory_items, vectorizer, X, nn, COMMERCIAL_SIGNALS
    catalog = get_catalog()
    inventory_items = [row["product_name"] for row in catalog]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    if inventory_items:
        X = vectorizer.fit_transform(inventory_items)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(X)
    else:
        X = None
        nn = None
    COMMERCIAL_SIGNALS = list(dict.fromkeys(COMMERCIAL_SIGNALS + [
        word.lower() for item in inventory_items for word in item.split() if len(word) > 3
    ]))


def upsert_product(product_name: str, buying_price: float, selling_price: float,
                   total_stock: int, reorder_level: int = DEFAULT_REORDER_LEVEL) -> None:
    """Owner API for adding a model or updating its catalogue values."""
    name = product_name.strip()
    if not name:
        raise ValueError("product_name is required")
    with db_conn() as conn:
        conn.execute("""INSERT INTO inventory
            (product_name, buying_price, selling_price, total_stock, total_sales, reorder_level, low_stock_alerted, active)
            VALUES (?, ?, ?, ?, 0, ?, 0, 1)
            ON CONFLICT(product_name) DO UPDATE SET
              buying_price=excluded.buying_price, selling_price=excluded.selling_price,
              total_stock=excluded.total_stock, reorder_level=excluded.reorder_level, active=1""",
            (name, float(buying_price), float(selling_price), int(total_stock), int(reorder_level)))
    refresh_catalog_models()


def update_stock(product_name: str, quantity: int, mode: str = "set") -> None:
    """Owner API for stock updates. mode is set, add, or subtract."""
    if mode not in {"set", "add", "subtract"}:
        raise ValueError("mode must be set, add, or subtract")
    op = {"set": "?", "add": "total_stock + ?", "subtract": "total_stock - ?"}[mode]
    with db_conn() as conn:
        cur = conn.execute(f"UPDATE inventory SET total_stock=MAX(0, {op}), low_stock_alerted=0 WHERE product_name=? COLLATE NOCASE",
                           (int(quantity), product_name))
        if cur.rowcount == 0:
            raise ValueError(f"Unknown product: {product_name}")
    refresh_catalog_models()


def verify_order_stock(items: List[dict], bulk_threshold: int = 10) -> dict:
    """Verify every requested line against a consistent SQL snapshot."""
    requested = []
    shortages = []
    total_units = 0
    with db_conn() as conn:
        for item in items or []:
            name = str(item.get("product", "")).strip()
            qty = max(1, int(item.get("quantity", 1) or 1))
            row = conn.execute("SELECT * FROM inventory WHERE product_name=? COLLATE NOCASE AND COALESCE(active,1)=1", (name,)).fetchone()
            current = dict(row) if row else None
            stock = int(current["total_stock"]) if current else 0
            price = float(current["selling_price"]) if current else None
            requested.append({"product": current["product_name"] if current else name, "quantity": qty,
                              "price": price, "stock": stock, "available": bool(current and stock >= qty)})
            total_units += qty
            if not current or stock < qty:
                shortages.append({"product": name, "requested": qty, "available": stock,
                                  "status": "model not found" if not current else "shortage"})
    return {"items": requested, "shortages": shortages, "total_units": total_units,
            "is_bulk": total_units >= bulk_threshold,
            "requires_owner": bool(shortages) or total_units >= bulk_threshold,
            "all_available": not shortages}


def get_last_uid() -> int:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM bot_settings WHERE key='last_uid'").fetchone()
        return int(row[0]) if row else 0


def set_last_uid(uid: int):
    with db_conn() as conn:
        conn.execute("INSERT INTO bot_settings (key, value) VALUES ('last_uid', ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(uid),))


def already_processed(message_id: str) -> bool:
    if not message_id:
        return False
    with db_conn() as conn:
        return conn.execute("SELECT 1 FROM processed_messages WHERE message_id=?",
                            (message_id,)).fetchone() is not None


def mark_processed(message_id: str):
    if not message_id:
        return
    with db_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO processed_messages VALUES (?, ?)",
                     (message_id, datetime.now().isoformat(timespec="seconds")))


def get_client_status(email_addr: str) -> Optional[dict]:
    if not email_addr:
        return None
    e = email_addr.lower()
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE email=?", (e,)).fetchone()
        if not row:
            return None
        data = dict(row)
        items = conn.execute("SELECT * FROM order_items WHERE order_email=?", (e,)).fetchall()
        data["Requested Items"] = [{
            "product": i["product_name"], "quantity": i["quantity"],
            "price": i["price_each"], "discount": i["discount_percent"] or 0.0,
        } for i in items]
        return data


def update_client_status(email_addr: str, client_name: str, requested_items: list,
                         new_status: str, retain_items: bool = True):
    e = email_addr.lower()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_conn() as conn:
        c = conn.cursor()
        existing = c.execute("SELECT quotation_sent, lpo_sent, invoice_sent "
                             "FROM orders WHERE email=?", (e,)).fetchone()
        q = f"Yes - {now}" if new_status == "QUOTE_SENT" else (existing[0] if existing else "No")
        l = f"Yes - {now}" if new_status == "LPO_SENT" else (existing[1] if existing else "No")
        i = f"Yes - {now}" if new_status == "INVOICE_SENT" else (existing[2] if existing else "No")

        c.execute("""INSERT INTO orders (email, client_name, status, last_updated,
                                         quotation_sent, lpo_sent, invoice_sent)
                     VALUES (?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(email) DO UPDATE SET
                        client_name=excluded.client_name, status=excluded.status,
                        last_updated=excluded.last_updated,
                        quotation_sent=excluded.quotation_sent,
                        lpo_sent=excluded.lpo_sent, invoice_sent=excluded.invoice_sent""",
                  (e, client_name, new_status, now, q, l, i))

        if requested_items and not retain_items:
            c.execute("DELETE FROM order_items WHERE order_email=?", (e,))
            for item in requested_items:
                product = item.get("product", "DJI Neo 2")
                qty = int(item.get("quantity", 1) or 1)
                inv = c.execute("SELECT product_name, selling_price FROM inventory "
                                "WHERE product_name=? COLLATE NOCASE", (product,)).fetchone()
                name = inv["product_name"] if inv else product
                if not inv and not item.get("price"):
                    raise ValueError(f"Cannot price unknown SQL catalogue item: {product}")
                price = float(item.get("price") or inv["selling_price"])
                disc = float(item.get("discount", 0.0) or 0.0)
                c.execute("INSERT INTO order_items (order_email, product_name, quantity,"
                          " price_each, discount_percent) VALUES (?, ?, ?, ?, ?)",
                          (e, name, qty, price, disc))


def record_sale(order_email: str, client_name: str, invoice_ref: str) -> int:
    """Record invoice lines and deduct stock atomically, with a final availability check."""
    e = order_email.lower()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    with db_conn() as conn:
        items = conn.execute("SELECT * FROM order_items WHERE order_email=?", (e,)).fetchall()
        if not items:
            raise ValueError(f"No order items found for {e}")
        for it in items:
            qty = int(it["quantity"] or 0)
            inv = conn.execute("SELECT * FROM inventory WHERE product_name=? COLLATE NOCASE AND COALESCE(active,1)=1", (it["product_name"],)).fetchone()
            if not inv or int(inv["total_stock"]) < qty:
                raise ValueError(f"Insufficient stock for {it['product_name']}: requested {qty}, available {int(inv['total_stock']) if inv else 0}")
            unit = float(it["price_each"] or inv["selling_price"])
            disc = float(it["discount_percent"] or 0.0)
            cost = float(inv["buying_price"])
            net = round(qty * unit * (1 - disc / 100.0), 2)
            rows.append((e, client_name, it["product_name"], qty, unit, disc, cost, net,
                         round(net - qty * cost, 2), invoice_ref, now, 1))
            updated = conn.execute("UPDATE inventory SET total_stock=total_stock-?, total_sales=total_sales+? WHERE product_name=? COLLATE NOCASE AND total_stock>=?",
                                   (qty, qty, it["product_name"], qty))
            if updated.rowcount != 1:
                raise ValueError(f"Stock changed while finalizing {it['product_name']}; invoice not applied")
        conn.executemany("""INSERT INTO sales (order_email, client_name, product_name, quantity, unit_price, discount_percent, buying_price, net_revenue, gross_profit, invoice_ref, sold_at, stock_applied) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    logging.info(f"LEDGER: {len(rows)} line(s) recorded and stock deducted for {e} (ref {invoice_ref}).")
    return len(rows)



def owner_metrics() -> dict:
    with db_conn() as conn:
        c = conn.cursor()
        today = c.execute("""SELECT COALESCE(SUM(quantity),0), COALESCE(SUM(net_revenue),0),
                                    COALESCE(SUM(gross_profit),0) FROM sales
                             WHERE date(sold_at)=date('now','localtime')""").fetchone()
        alltime = c.execute("""SELECT COALESCE(SUM(quantity),0), COALESCE(SUM(net_revenue),0),
                                      COALESCE(SUM(gross_profit),0) FROM sales""").fetchone()
        stock = c.execute("SELECT COALESCE(SUM(total_stock),0) FROM inventory").fetchone()[0]
        open_deals = c.execute("""SELECT COUNT(*) FROM orders WHERE status IN
                                  ('QUOTE_SENT','AWAITING_DISCOUNT',
                                   'AWAITING_CLARIFICATION','LPO_SENT')""").fetchone()[0]
    return {"today_qty": today[0], "today_revenue": today[1], "today_profit": today[2],
            "total_qty": alltime[0], "total_revenue": alltime[1], "total_profit": alltime[2],
            "stock": stock, "open_deals": open_deals}


# =============================================================================
# 4b. OWNER REPORTING ENGINE  (v5.2)
# =============================================================================
WINDOWS = {
    "today":      ("date(sold_at) = date('now','localtime')", "today"),
    "yesterday":  ("date(sold_at) = date('now','localtime','-1 day')", "yesterday"),
    "week":       ("date(sold_at) >= date('now','localtime','-6 days')", "the last 7 days"),
    "month":      ("strftime('%Y-%m', sold_at) = strftime('%Y-%m','now','localtime')", "this month"),
    "last_month": ("strftime('%Y-%m', sold_at) = strftime('%Y-%m','now','localtime','-1 month')",
                   "last month"),
    "year":       ("strftime('%Y', sold_at) = strftime('%Y','now','localtime')", "this year"),
    "all":        ("1=1", "all time"),
}


def detect_window(text: str) -> str:
    if has_words(text, ["today"]):                                  return "today"
    if has_words(text, ["yesterday"]):                              return "yesterday"
    if has_words(text, ["last month", "previous month"]):           return "last_month"
    if has_words(text, ["this month", "month", "monthly", "mtd"]):  return "month"
    if has_words(text, ["this week", "week", "weekly", "7 days"]):  return "week"
    if has_words(text, ["this year", "year", "ytd", "annual"]):     return "year"
    return "all"


def detect_report(text: str) -> str:
    if has_words(text, ["low stock", "running out", "run out", "reorder", "restock",
                        "out of stock", "running low", "short"]):
        return "low_stock"
    if has_words(text, ["stock", "inventory", "remaining", "left", "on hand", "in hand"]):
        return "stock"
    if has_words(text, ["pipeline", "open deals", "pending", "quotes out", "outstanding"]):
        return "pipeline"
    return "sales"


def sales_rows(window_key: str) -> List[dict]:
    where, _ = WINDOWS.get(window_key, WINDOWS["all"])
    with db_conn() as conn:
        return [dict(r) for r in conn.execute(f"""
            SELECT product_name,
                   SUM(quantity)               AS units,
                   SUM(net_revenue)            AS revenue,
                   SUM(gross_profit)           AS profit,
                   COUNT(DISTINCT order_email) AS customers,
                   MAX(discount_percent)       AS max_discount
            FROM sales WHERE {where}
            GROUP BY product_name COLLATE NOCASE
            ORDER BY units DESC, revenue DESC""").fetchall()]


def stock_rows() -> List[dict]:
    with db_conn() as conn:
        rows = conn.execute(f"""
            SELECT i.product_name, i.total_stock, i.total_sales,
                   COALESCE(i.reorder_level, {DEFAULT_REORDER_LEVEL}) AS reorder_level,
                   COALESCE((SELECT SUM(s.quantity) FROM sales s
                             WHERE s.product_name = i.product_name COLLATE NOCASE
                               AND date(s.sold_at) >= date('now','localtime',
                                    '-{VELOCITY_WINDOW_DAYS - 1} days')), 0) AS units_recent
            FROM inventory i ORDER BY i.product_name""").fetchall()

    out = []
    for r in rows:
        velocity = r["units_recent"] / float(VELOCITY_WINDOW_DAYS)
        days_left = round(r["total_stock"] / velocity, 1) if velocity > 0 else None
        out.append({
            "product": r["product_name"], "stock": r["total_stock"],
            "reorder_level": r["reorder_level"], "sold": r["total_sales"],
            "velocity": round(velocity, 3), "days_left": days_left,
            "low": r["total_stock"] <= r["reorder_level"],
            "urgent": days_left is not None and days_left <= LOW_STOCK_URGENT_DAYS,
        })
    return out


def pipeline_rows() -> List[dict]:
    with db_conn() as conn:
        return [dict(r) for r in conn.execute("""
            SELECT o.email, o.client_name, o.status, o.last_updated,
                   COALESCE(SUM(oi.quantity), 0) AS units,
                   COALESCE(SUM(oi.quantity * oi.price_each *
                                (1 - oi.discount_percent / 100.0)), 0) AS value
            FROM orders o LEFT JOIN order_items oi ON oi.order_email = o.email
            WHERE o.status IN ('QUOTE_SENT','AWAITING_DISCOUNT',
                               'AWAITING_CLARIFICATION','LPO_SENT')
            GROUP BY o.email ORDER BY value DESC""").fetchall()]


def _html_table(headers: List[str], rows: List[list], aligns: List[str] = None) -> str:
    aligns = aligns or ["left"] * len(headers)
    th = "".join(f"<th style='padding:8px 10px;text-align:{a};font-size:12px;"
                 f"text-transform:uppercase;letter-spacing:.4px;color:#5a6b7b;"
                 f"border-bottom:2px solid #dfe6ec;'>{html.escape(str(h))}</th>"
                 for h, a in zip(headers, aligns))
    body = ""
    for i, row in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#f7fafc"
        tds = "".join(f"<td style='padding:9px 10px;text-align:{a};font-size:14px;"
                      f"border-bottom:1px solid #eef2f5;'>{cell}</td>"
                      for cell, a in zip(row, aligns))
        body += f"<tr style='background:{bg};'>{tds}</tr>"
    return (f"<table style='width:100%;border-collapse:collapse;margin:10px 0 18px;'>"
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")


def build_owner_report(question: str) -> Dict[str, str]:
    text = (question or "").lower()
    kind = detect_report(text)
    wkey = detect_window(text)
    wlabel = WINDOWS[wkey][1]
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")

    # ---------------------------------------------------------------- stock --
    if kind in ("stock", "low_stock"):
        rows = stock_rows()
        if kind == "low_stock":
            rows = [r for r in rows if r["low"]]
            title = "Low stock alert"
        else:
            title = "Stock on hand"

        if not rows:
            plain = ("Nothing is at or below its reorder level right now."
                     if kind == "low_stock" else "No inventory records found.")
            body_html = f"<p style='font-size:15px;color:#2f4256;'>{plain}</p>"
        else:
            lines, trows = [], []
            for r in rows:
                left = f"{r['days_left']:g} days" if r["days_left"] is not None else "no recent sales"
                flag = " [URGENT]" if r["urgent"] else (" [LOW]" if r["low"] else "")
                lines.append(f"- {r['product']}: {r['stock']} in stock "
                             f"(reorder at {r['reorder_level']}), ~{left} left{flag}")
                colour = "#c0392b" if r["urgent"] else ("#e67e22" if r["low"] else "#2f4256")
                trows.append([
                    f"<strong>{html.escape(r['product'])}</strong>",
                    f"<span style='color:{colour};font-weight:600;'>{r['stock']}</span>",
                    str(r["reorder_level"]),
                    f"{r['velocity']:g}/day",
                    f"<span style='color:{colour};'>{left}</span>"])
            plain = "\n".join(lines)
            body_html = _html_table(
                ["Model", "In stock", "Reorder at", "Velocity", "Runs out in"],
                trows, ["left", "right", "right", "right", "right"])

        subject = f"Aerotech - {title}"
        plain = f"{title.upper()} ({stamp})\n\n{plain}"

    # ------------------------------------------------------------- pipeline --
    elif kind == "pipeline":
        rows = pipeline_rows()
        total = sum(r["value"] for r in rows)
        subject = f"Aerotech - Open pipeline ({len(rows)} deals)"
        if not rows:
            plain = f"OPEN PIPELINE ({stamp})\n\nNo open deals."
            body_html = "<p style='font-size:15px;color:#2f4256;'>No open deals.</p>"
        else:
            lines = [f"- {r['client_name'] or r['email']} ({r['email']}): "
                     f"{r['status']}, {r['units']} units, {money(r['value'])}" for r in rows]
            plain = (f"OPEN PIPELINE ({stamp})\n\n{len(rows)} deals worth "
                     f"{money(total)}\n\n" + "\n".join(lines))
            body_html = _html_table(
                ["Client", "Stage", "Units", "Value"],
                [[f"<strong>{html.escape(r['client_name'] or r['email'])}</strong>"
                  f"<br><span style='font-size:12px;color:#7b8b9a;'>{html.escape(r['email'])}</span>",
                  r["status"].replace("_", " ").title(), str(r["units"]),
                  money(r["value"])] for r in rows],
                ["left", "left", "right", "right"])

    # ---------------------------------------------------------------- sales --
    else:
        rows = sales_rows(wkey)
        units = sum(r["units"] for r in rows)
        revenue = sum(r["revenue"] for r in rows)
        profit = sum(r["profit"] for r in rows)
        subject = (f"Aerotech - {units} {'drone' if units == 1 else 'drones'} sold {wlabel}")

        if not rows:
            plain = f"SALES - {wlabel.upper()} ({stamp})\n\nNo drones sold {wlabel}."
            body_html = (f"<p style='font-size:15px;color:#2f4256;'>"
                         f"No drones sold {html.escape(wlabel)}.</p>")
        else:
            lines = [f"- {r['product_name']}: {r['units']} "
                     f"{'unit' if r['units'] == 1 else 'units'}, {money(r['revenue'])} "
                     f"revenue, {money(r['profit'])} profit" for r in rows]
            plain = (f"SALES - {wlabel.upper()} ({stamp})\n\n"
                     f"TOTAL: {units} drones | {money(revenue)} revenue | "
                     f"{money(profit)} gross profit\n\nBY MODEL:\n" + "\n".join(lines))
            body_html = _html_table(
                ["Model", "Units", "Revenue", "Gross profit"],
                [[f"<strong>{html.escape(r['product_name'])}</strong>", str(r["units"]),
                  money(r["revenue"]), money(r["profit"])] for r in rows],
                ["left", "right", "right", "right"])
            kpis = [("Drones sold", str(units)), ("Revenue", money(revenue)),
                    ("Gross profit", money(profit))]
            cards = "".join(
                f"<td style='padding:12px 10px;background:#f2f7fb;border-radius:8px;"
                f"text-align:center;'>"
                f"<div style='font-size:11px;color:#7b8b9a;text-transform:uppercase;"
                f"letter-spacing:.5px;'>{k}</div>"
                f"<div style='font-size:19px;font-weight:700;color:#1b4f72;"
                f"padding-top:4px;'>{v}</div></td>"
                f"<td style='width:8px;'></td>" for k, v in kpis)
            body_html = (f"<table style='width:100%;border-collapse:separate;"
                         f"border-spacing:0;margin-bottom:6px;'><tr>{cards}</tr></table>"
                         + body_html)

    full_html = f"""<div style="font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
         max-width:640px;margin:0 auto;padding:16px;color:#2f4256;">
      <div style="border-bottom:3px solid #1b4f72;padding-bottom:10px;margin-bottom:16px;">
        <div style="font-size:19px;font-weight:700;color:#1b4f72;">AEROTECH DRONES</div>
        <div style="font-size:13px;color:#7b8b9a;">{html.escape(subject.replace('Aerotech - ', ''))}
          &middot; {stamp}</div>
      </div>
      {body_html}
      <div style="font-size:11px;color:#9aa8b5;border-top:1px solid #eef2f5;
                  padding-top:10px;margin-top:6px;">
        Figures read directly from the sales ledger. Reply 'send report' for the
        full interactive dashboard.
      </div></div>"""

    return {"subject": subject, "plain": plain, "html": full_html}


def check_low_stock() -> int:
    rows = stock_rows()
    with db_conn() as conn:
        flags = {r["product_name"]: (r["low_stock_alerted"] or 0) for r in
                 conn.execute("SELECT product_name, low_stock_alerted FROM inventory")}

    firing = [r for r in rows if r["low"] and not flags.get(r["product"])]
    rearm = [r["product"] for r in rows if not r["low"] and flags.get(r["product"])]

    with db_conn() as conn:
        for r in firing:
            conn.execute("UPDATE inventory SET low_stock_alerted=1 WHERE product_name=?",
                         (r["product"],))
        for p in rearm:
            conn.execute("UPDATE inventory SET low_stock_alerted=0 WHERE product_name=?", (p,))
    if rearm:
        logging.info(f"Low-stock alert re-armed for: {', '.join(rearm)}")
    if not firing:
        return 0

    lines, trows = [], []
    for r in firing:
        left = f"{r['days_left']:g} days" if r["days_left"] is not None else "no recent sales"
        lines.append(f"- {r['product']}: only {r['stock']} left "
                     f"(reorder level {r['reorder_level']}) - about {left} of cover")
        colour = "#c0392b" if r["urgent"] else "#e67e22"
        trows.append([f"<strong>{html.escape(r['product'])}</strong>",
                      f"<span style='color:{colour};font-weight:700;'>{r['stock']}</span>",
                      str(r["reorder_level"]), f"{r['velocity']:g}/day",
                      f"<span style='color:{colour};'>{left}</span>"])

    plain = ("LOW STOCK - ACTION NEEDED\n\n" + "\n".join(lines) +
             "\n\nForecast is units sold per day over the last "
             f"{VELOCITY_WINDOW_DAYS} days. Restock before these run out.")
    body = _html_table(["Model", "Left", "Reorder at", "Velocity", "Cover"],
                       trows, ["left", "right", "right", "right", "right"])
    html_body = f"""<div style="font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
        max-width:640px;margin:0 auto;padding:16px;color:#2f4256;">
      <div style="background:#fdecea;border-left:4px solid #c0392b;padding:12px 14px;
                  border-radius:4px;margin-bottom:14px;">
        <div style="font-size:16px;font-weight:700;color:#c0392b;">Low stock - action needed</div>
        <div style="font-size:13px;color:#7b4a45;">
          {len(firing)} model(s) at or below reorder level</div></div>
      {body}
      <div style="font-size:11px;color:#9aa8b5;">Cover is projected from units sold
        per day over the last {VELOCITY_WINDOW_DAYS} days.</div></div>"""

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = EMAIL_USER, EMAIL_USER
    msg["X-Drone-Bot"] = "true"
    msg["Auto-Submitted"] = "auto-replied"
    msg["Subject"] = f"LOW STOCK: {len(firing)} model(s) need restocking"
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    send_email_sync(msg)
    logging.warning(f"LOW STOCK alert sent for: {', '.join(r['product'] for r in firing)}")
    return len(firing)


initialize_database()
refresh_catalog_models()


# =============================================================================
# 5. LOCAL LLM
# =============================================================================
logging.info("Loading Qwen2.5-1.5B GGUF (CPU)...")
model_path = hf_hub_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                             filename="qwen2.5-1.5b-instruct-q4_k_m.gguf")
local_llm = LlamaCpp(model_path=model_path, temperature=0.1, max_tokens=256,
                     n_ctx=2048, n_threads=max(os.cpu_count() - 1, 1), verbose=False)

ensure_trace_schema(DB_FILE)
# Fixed parameter set for live monitor visualization
export_models(inventory_items, vectorizer, X, spam_vectorizer,
              spam_classifier, KNN_MAX_DISTANCE,
              training_emails, email_labels, model_path)


# =============================================================================
# 6. STATE
# =============================================================================
class AgentState(TypedDict):
    email_id: str
    message_id: str
    sender_email: str
    display_name: str
    email_subject: str
    email_body: str
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed",
                    "invoice_response", "request_discount", "owner_discount_decision",
                    "owner_query", "unrelated", "ask_clarification", "human_handoff"]
    company_name: Optional[str]
    requested_items: List[Dict[str, Any]]
    feedback_reason: Optional[str]
    target_client_email: Optional[str]
    generated_doc_path: Optional[str]
    doc_type_sent: Optional[str]
    reply_message: Optional[str]
    reply_html: Optional[str]
    reply_subject: Optional[str]
    retain_items: Optional[bool]
    handoff_note: Optional[str]
    error_message: Optional[str]

CONTINUATION_INTENTS = {"quote_approval", "delivery_confirmed", "invoice_response"}

# Last raw model output, captured so the monitor can show what Qwen really said.
LAST_LLM_RAW = {"v": ""}


# =============================================================================
# 7. EXTRACTION NODE
# =============================================================================
async def extract_requirements(state: AgentState) -> dict:
    sender = (state.get("sender_email") or "").lower()
    subject = state.get("email_subject") or ""
    body_clean = clean_body(state.get("email_body") or "")
    text = body_clean.lower()
    display = state.get("display_name") or (sender.split("@")[0] if sender else "Valued Client")

    # --- 1. OWNER OVERRIDES ------------------------------------------------
    if sender == EMAIL_USER.lower():
        if "DISCOUNT REQUEST:" in subject:
            return {"intent": "owner_discount_decision",
                    "target_client_email": subject.split("DISCOUNT REQUEST:")[-1].strip(),
                    "company_name": "Owner", "requested_items": [], "feedback_reason": ""}
        return {"intent": "owner_query", "company_name": "Aerotech Drones Owner",
                "requested_items": [], "feedback_reason": ""}

    db_record = get_client_status(sender)
    current_status = db_record.get("status") if db_record else None

    # --- 2. SPAM FILTER ----------------------------------------------------
    ml_says_spam = spam_classifier.predict(spam_vectorizer.transform([text]))[0] == "spam"
    has_signal = has_words(text, COMMERCIAL_SIGNALS, respect_negation=False) or \
                 has_words(subject.lower(), COMMERCIAL_SIGNALS, respect_negation=False)
    if ml_says_spam and not has_signal and current_status is None:
        logging.info(f"SPAM: dropped mail from {sender}")
        return {"intent": "unrelated", "company_name": display, "requested_items": []}

    # --- 3. CLARIFICATION MEMORY -------------------------------------------
    if current_status == "AWAITING_CLARIFICATION":
        if has_words(text, ["yes", "correct", "right", "yep", "yeah", "sure",
                            "proceed", "confirm", "confirmed"]):
            return {"intent": "new_inquiry", "company_name": display,
                    "requested_items": db_record["Requested Items"], "feedback_reason": ""}

    # --- 4. LLM EXTRACTION -------------------------------------------------
    prompt = PromptTemplate(
        template="""<|im_start|>system
You are an intent and product extractor. Extract the EXACT product words the user typed without changing or correcting them.
Allowed intents: new_inquiry, quote_approval, request_discount.
<|im_end|>
<|im_start|>user
Email: "{body}"
Extract requested drones into items array keeping user's exact words for product. Also extract "quantity" as an integer if specified.<|im_end|>
<|im_start|>assistant
{{
  "intent":""",
        input_variables=["body"])

    intent, items_data = "new_inquiry", []
    try:
        raw = await asyncio.to_thread(local_llm.invoke, prompt.format(body=body_clean))
        logging.info(f"LLM raw -> {raw[:200]!r}")          
        LAST_LLM_RAW["v"] = raw
        candidate = ('{\n  "intent":' + raw).replace("```", "").strip()
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            intent = parsed.get("intent", "new_inquiry")
            items_data = parsed.get("items") or parsed.get("product") or parsed.get("products") or []
            
            # --- LLM Safety Net ---
            if isinstance(items_data, str):
                items_data = [{"product": items_data}]
            elif isinstance(items_data, dict):
                # Catch nested "items" lists inside a dictionary
                if "items" in items_data and  isinstance(items_data["items"],list):
                    items_data = items_data["items"]
                else:
                    items_data = [items_data]
                
    except Exception as e:
        logging.warning(f"LLM extraction unusable ({e}); falling back to keywords.")

    # --- 4b. DETERMINISTIC CATALOGUE PRE-PASS ------------------------------
    regex_items = scan_products_in_text(body_clean)
    if not items_data and regex_items:
        items_data = regex_items
        logging.info(f"LLM returned nothing; catalogue scan recovered {regex_items}")

    # --- 5. PIPELINE STATUS OVERRIDES (MOVED UP) ---------------------------
    if current_status == "QUOTE_SENT":
        if has_words(text, ["discount", "offer", "reduce", "lower", "cheaper", "best price"]):
            intent = "request_discount"
        elif has_words(text, ["approve", "approved", "proceed", "confirm", "confirmed",
                              "accept", "accepted", "go ahead", "lpo", "purchase order",
                              "agree", "agreed", "yes"]):
            intent = "quote_approval"
        elif has_words(text, ["quote", "quotation", "revise", "revised", "amend"]):
            intent = "new_inquiry"
        else:
            intent = "human_handoff"

    elif current_status == "LPO_SENT":
        if has_words(text, ["delivered", "delivery", "received", "invoice", "bill",
                            "completed", "handover", "installed"]):
            intent = "delivery_confirmed"
        elif has_words(text, ["discount", "reduce", "lower"]):
            intent = "request_discount"
        else:
            intent = "human_handoff"

    elif current_status == "AWAITING_DISCOUNT":
        intent = "human_handoff"

    else:
        if has_words(text, ["quote", "quotation", "price", "pricing", "cost", "buy",
                            "purchase", "order", "interested", "enquiry", "inquiry"]):
            intent = "new_inquiry"

    # --- 6. EXACT MATCH, THEN KNN CLARIFICATION ----------------------------
    verified_items, clarifications = [], []
    
    # If this is a continuation (like LPO approval), ignore LLM hallucinations
    if intent in CONTINUATION_INTENTS or intent == "request_discount":
        items_data = []

    for item in items_data:
        name = (item.get("product") or item.get("name") or "").strip()
        if not name:
            continue
            
        # Capture the quantity immediately
        qty = int(item.get("quantity", 1) or 1)
        
        exact = next((inv for inv in inventory_items if inv.lower() == name.lower()), None)
        if exact:
            verified_items.append({"product": exact, "quantity": qty})
            continue
        distances, indices = nn.kneighbors(vectorizer.transform([name]))
        best = inventory_items[indices[0][0]] if nn is not None and distances[0][0] < KNN_MAX_DISTANCE else None
        
        # Pass the requested quantity into the clarification list
        clarifications.append((name, best, qty))

    if clarifications and regex_items:
        logging.info(f"Clarification suppressed; catalogue scan matched "
                     f"{[i['product'] for i in regex_items]}")
        verified_items, clarifications = regex_items, []

    if clarifications:
        lines = [f"Hello {display},", "",
                 "Thanks for your enquiry. We need one quick clarification before "
                 "we can raise your quotation:", ""]
        guessed = []
        # Unpack the quantity here
        for original, guess, qty in clarifications:
            if guess:
                lines.append(f"- You asked for '{original}'. Did you mean {guess}?")
                # Apply the original quantity instead of hardcoding '1'
                guessed.append({"product": guess, "quantity": qty})
            else:
                lines.append(f"- We could not match '{original}' to our catalogue. "
                             f"Could you confirm the exact model?")
        lines += ["", "Just reply to this email to confirm and we will send the "
                      "official quotation straight away.", "",
                  "Best Regards,", "Aerotech Drones"]
        return {"intent": "ask_clarification", "company_name": display,
                "requested_items": guessed, "reply_message": "\n".join(lines),
                "doc_type_sent": "Clarification"}

    # --- 7. WHICH LINE ITEMS CARRY FORWARD? --------------------------------
    if intent in CONTINUATION_INTENTS and db_record and db_record["Requested Items"]:
        verified_items = db_record["Requested Items"]
    elif not verified_items and db_record and db_record["Requested Items"]:
        verified_items = db_record["Requested Items"]

    if not verified_items and intent == "new_inquiry":
        return {"intent": "human_handoff", "company_name": display, "requested_items": [],
                "handoff_note": "Priced enquiry with no identifiable product."}

    return {"intent": intent, "company_name": display,
            "requested_items": verified_items, "feedback_reason": ""}


# =============================================================================
# 8. ACTION NODES
# =============================================================================
def route_workflow(state: AgentState) -> str:
    return {
        "new_inquiry": "check_stock",
        "request_discount": "ask_owner_discount",
        "owner_discount_decision": "apply_discount_and_requote",
        "quote_approval": "generate_lpo",
        "delivery_confirmed": "generate_invoice",
        "invoice_response": "generate_invoice",
        "owner_query": "answer_owner",
        "ask_clarification": "dispatch",
        "human_handoff": "human_handoff",
    }.get(state["intent"], "ignore_email")


def check_stock(state: AgentState) -> dict:
    check = verify_order_stock(state.get("requested_items") or [])
    if check["requires_owner"]:
        return {"intent": "request_discount", "requested_items": check["items"],
                "feedback_reason": "bulk_or_shortage"}
    return {"intent": "new_inquiry", "requested_items": check["items"]}


def ignore_email(state: AgentState) -> dict:
    logging.info(f"IGNORED: {state.get('sender_email')}")
    return {"doc_type_sent": "Ignored"}


def human_handoff(state: AgentState) -> dict:
    rec = get_client_status(state.get("sender_email"))
    note = state.get("handoff_note") or "Bot could not confidently classify this message."
    reply = (f"{note}\n\n"
             f"From    : {state.get('display_name')} <{state.get('sender_email')}>\n"
             f"Status  : {rec.get('status') if rec else 'no open deal'}\n"
             f"Subject : {state.get('email_subject')}\n\n"
             f"--- Message ---\n{(state.get('email_body') or '')[:2000]}\n\n"
             f"No automated reply was sent. Please respond manually.")
    logging.info(f"HANDOFF to owner for {state.get('sender_email')}")
    return {"doc_type_sent": "Human Handoff", "target_client_email": EMAIL_USER,
            "reply_message": reply}


def ask_owner_discount(state: AgentState) -> dict:
    check = verify_order_stock(state.get("requested_items") or [])
    update_client_status(state["sender_email"], state["company_name"],
                         check["items"], "AWAITING_DISCOUNT", retain_items=False)
    lines = []
    for item in check["items"]:
        status = f"{item['stock']} in stock" if item["available"] else f"SHORTAGE - {item['stock']} in stock"
        lines.append(f"- {item['quantity']} x {item['product']} - {status} - {money(item['price'] or 0)} each")
    reason = []
    if check["is_bulk"]:
        reason.append(f"bulk request: {check['total_units']} total units")
    if check["shortages"]:
        reason.append("one or more requested quantities exceed current stock")
    return {"doc_type_sent": "Internal Owner Alert", "target_client_email": EMAIL_USER,
            "reply_message": (f"Name: {state.get('display_name')}\nEmail: {state['sender_email']}\n\n"
                              f"Requested items ({'; '.join(reason)}):\n" + "\n".join(lines) +
                              "\n\nReply with APPROVE and a discount, for example 'APPROVE 10% TOTAL' or 'APPROVE 10% Mavic 4 Pro'." )}


def apply_discount_and_requote(state: AgentState) -> dict:
    target = state.get("target_client_email") or ""
    body = clean_body(state.get("email_body") or "")
    approved = bool(re.search(r"\bapprove(?:d)?\b", body, re.I))
    discount = parse_discount(body)
    if not approved:
        return {"doc_type_sent": "Internal Owner Alert", "target_client_email": EMAIL_USER,
                "reply_message": "Approval not detected. Reply with APPROVE followed by a percentage discount."}
    rec = get_client_status(target)
    if not rec:
        return {"doc_type_sent": "Internal Owner Alert", "target_client_email": EMAIL_USER,
                "reply_message": f"No open order found for {target}."}
    text_lower = body.lower()
    item_specific = None
    for item in rec["Requested Items"]:
        if item["product"].lower() in text_lower or re.sub(r"^dji\s+", "", item["product"], flags=re.I).lower() in text_lower:
            item_specific = item["product"]
            break
    with db_conn() as conn:
        if item_specific and "total" not in text_lower and "grand" not in text_lower:
            conn.execute("UPDATE order_items SET discount_percent=? WHERE order_email=? AND product_name=? COLLATE NOCASE", (discount, target.lower(), item_specific))
        else:
            conn.execute("UPDATE order_items SET discount_percent=? WHERE order_email=?", (discount, target.lower()))
    rec = get_client_status(target)
    return {"sender_email": target, "company_name": rec["client_name"] or "Valued Client",
            "requested_items": rec["Requested Items"], "target_client_email": target, "retain_items": True,
            "reply_message": f"Management approved a {discount:g}% discount. Please find the updated quotation attached."}


async def answer_owner(state: AgentState) -> dict:
    question = clean_body(state.get("email_body") or "") or (state.get("email_subject") or "")
    report = await asyncio.to_thread(build_owner_report, question)
    logging.info(f"OWNER REPORT: {report['subject']}")

    intro = ""
    try:
        p = ("<|im_start|>system\nWrite ONE short sentence introducing a sales report. "
             "Do NOT include any numbers, figures, statistics or dates - the report "
             "itself follows separately. Maximum 15 words.<|im_end|>\n"
             f"<|im_start|>user\nThe owner asked: {question}<|im_end|>\n"
             "<|im_start|>assistant\n")
        cand = (await asyncio.to_thread(local_llm.invoke, p)).strip().split("\n")[0].strip()
        if cand and len(cand) <= 140 and not re.search(r"\d", cand):
            intro = cand + "\n\n"
    except Exception as e:
        logging.warning(f"Owner intro LLM failed ({e}); sending figures only.")

    return {"doc_type_sent": "Owner Reply", "target_client_email": EMAIL_USER,
            "reply_subject": report["subject"],
            "reply_message": intro + report["plain"],
            "reply_html": report["html"]}


# =============================================================================
# 9. PDF GENERATION
# =============================================================================
def generate_professional_pdf(doc_type: str, state: AgentState) -> str:
    client_email = state.get("sender_email")
    client_name = state.get("company_name") or "Valued Client"

    state_items = state.get("requested_items") or []
    db_items = (get_client_status(client_email) or {}).get("Requested Items") or []
    items = state_items or db_items

    subtotal, savings, top_discount, rows_html = 0.0, 0.0, 0.0, ""
    with db_conn() as conn:
        c = conn.cursor()
        for idx, item in enumerate(items, start=1):
            product = item.get("product", "DJI Neo 2")
            qty = int(item.get("quantity", 1) or 1)
            disc = float(item.get("discount", 0.0) or 0.0)
            top_discount = max(top_discount, disc)

            price = item.get("price")
            if not price:
                inv = c.execute("SELECT selling_price FROM inventory "
                                "WHERE product_name=? COLLATE NOCASE", (product,)).fetchone()
                if not inv:
                    raise ValueError(f"Cannot price unknown SQL catalogue item: {product}")
                price = float(inv["selling_price"])
            price = float(price)

            line_total = qty * price
            subtotal += line_total
            savings += (price - price * (1 - disc / 100.0)) * qty

            rows_html += f"""
            <tr><td style="text-align:center;">{idx}</td>
                <td><strong>{html.escape(str(product))}</strong><br>
                    <span style="font-size:10px;color:#555;">Standard specifications apply.</span></td>
                <td style="text-align:center;">{qty}</td>
                <td style="text-align:right;">{money(price)}</td>
                <td style="text-align:center;">{disc:g}% off</td>
                <td style="text-align:right;">{money(price * (1 - disc / 100.0))}</td>
                <td style="text-align:right;">{money(line_total * (1 - disc / 100.0))}</td></tr>"""

    taxable = subtotal - savings
    tax = taxable * TAX_RATE
    grand_total = taxable + tax

    savings_html = ""
    if savings > 0:
        savings_html = (f"<tr><td><strong>Discount ({top_discount:g}%)</strong></td>"
                        f"<td style='text-align:right;color:#c0392b;'>"
                        f"<strong>-{money(savings)}</strong></td></tr>")

    css = """
        body { font-family:'Helvetica',sans-serif; color:#111; font-size:12px; margin:0; padding:20px; }
        .header-container { border-bottom:2px solid #000; padding-bottom:10px; margin-bottom:20px; }
        .company-title { font-size:24px; font-weight:bold; color:#2c3e50; margin:0; }
        .doc-title { font-size:20px; font-weight:bold; text-align:right; text-transform:uppercase; color:#555; }
        .info-grid { display:table; width:100%; margin-bottom:20px; }
        .info-col { display:table-cell; width:50%; vertical-align:top; }
        table { width:100%; border-collapse:collapse; margin-bottom:20px; }
        th { background-color:#f2f2f2; border:1px solid #000; padding:8px; font-size:11px; text-transform:uppercase; }
        td { border:1px solid #000; padding:8px; vertical-align:top; }
        .totals-table { width:45%; float:right; border-collapse:collapse; }
        .totals-table td { border:1px solid #000; padding:6px; }
        .totals-table .bold { font-weight:bold; background-color:#f9f9f9; }
    """
    run_id = uuid.uuid4().hex[:6].upper()
    doc_html = f"""<html><head><style>{css}</style></head><body>
      <div class='header-container'><table style='border:none;margin:0;padding:0;'><tr>
        <td style='border:none;padding:0;'><h1 class='company-title'>AEROTECH DRONES</h1></td>
        <td style='border:none;padding:0;' align='right'>
          <div class='doc-title'>{html.escape(doc_type)}</div></td></tr></table></div>
      <div class='info-grid'>
        <div class='info-col'><strong>Customer Name:</strong><br>{html.escape(client_name)}</div>
        <div class='info-col' style='text-align:right;'><strong>Ref:</strong> {run_id}<br>
          <strong>Date:</strong> {datetime.now().strftime('%d %b %Y')}</div></div>
      <table><tr><th width='5%'>S.NO</th><th width='45%'>ITEM DESCRIPTION</th>
        <th width='10%'>QTY</th><th width='15%'>ORIGINAL RATE</th><th width='10%'>DISCOUNT</th><th width='15%'>DISCOUNTED RATE</th><th width='15%'>AMOUNT</th></tr>
        {rows_html}</table>
      <table class='totals-table'>
        <tr><td>Subtotal</td><td style='text-align:right;'>{money(subtotal)}</td></tr>
        {savings_html}
        <tr><td>{TAX_LABEL} ({TAX_RATE * 100:g}%)</td><td style='text-align:right;'>{money(tax)}</td></tr>
        <tr class='bold'><td>Total Amount</td>
          <td style='text-align:right;'>{money(grand_total)}</td></tr></table>
      </body></html>"""

    os.makedirs("./docs", exist_ok=True)
    path = f"./docs/{doc_type.replace(' ', '_')}_{run_id}.pdf"
    HTML(string=doc_html).write_pdf(path)
    logging.info(f"PDF: {path} ({len(items)} line(s), total {money(grand_total)})")
    return path


async def generate_quote(state: AgentState) -> dict:
    return {"generated_doc_path": await asyncio.to_thread(generate_professional_pdf, "Quotation", state),
            "doc_type_sent": "Quotation"}


async def generate_lpo(state: AgentState) -> dict:
    return {"generated_doc_path": await asyncio.to_thread(generate_professional_pdf, "LPO", state),
            "doc_type_sent": "Local Purchase Order"}


async def generate_invoice(state: AgentState) -> dict:
    return {"generated_doc_path": await asyncio.to_thread(generate_professional_pdf, "Invoice", state),
            "doc_type_sent": "Tax Invoice",
            "reply_message": "Thank you. Please find your tax invoice attached."}

# =============================================================================
# 10. DISPATCH
# =============================================================================
def send_email_sync(msg: MIMEMultipart) -> bool:
    server = None
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        return True
    except Exception as e:
        logging.error(f"SMTP send failed: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


async def dispatch_and_update(state: AgentState) -> dict:
    doc_type = state.get("doc_type_sent", "Response")
    target = state.get("target_client_email") or state["sender_email"]

    msg = MIMEMultipart("alternative" if doc_type == "Owner Reply" else "mixed")
    msg["From"] = EMAIL_USER
    msg["To"] = target
    msg["X-Drone-Bot"] = "true"
    msg["Auto-Submitted"] = "auto-replied"

    if doc_type == "Internal Owner Alert":
        msg["Subject"] = f"DISCOUNT REQUEST: {state['sender_email']}"
        msg.attach(MIMEText(state["reply_message"], "plain"))
    elif doc_type == "Human Handoff":
        msg["Subject"] = f"NEEDS A HUMAN: {state.get('sender_email')}"
        msg.attach(MIMEText(state["reply_message"], "plain"))
    elif doc_type == "Owner Reply":
        msg["Subject"] = (state.get("reply_subject")
                          or f"Re: {state.get('email_subject') or 'Owner Query'}")
        msg.attach(MIMEText(state["reply_message"], "plain"))   
        if state.get("reply_html"):
            msg.attach(MIMEText(state["reply_html"], "html"))
    else:
        msg["Subject"] = "Re: Your inquiry - Aerotech Drones"
        msg.attach(MIMEText(state.get("reply_message") or
                            f"Dear {state.get('company_name') or 'Valued Client'},\n\n"
                            f"Please find your requested {doc_type} attached.\n\n"
                            f"Best Regards,\nAerotech Drones", "plain"))
        path = state.get("generated_doc_path")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(path))
            part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
            msg.attach(part)

    if not await asyncio.to_thread(send_email_sync, msg):
        logging.error(f"Status NOT advanced for {target} - send failed.")
        return {"error_message": "smtp_failed"}

    logging.info(f"SENT {doc_type} -> {target}")

    internal = doc_type in ("Owner Reply", "Internal Owner Alert", "Human Handoff", "Ignored")
    if target.lower() != EMAIL_USER.lower() and not internal:
        new_status = {"Quotation": "QUOTE_SENT", "Local Purchase Order": "LPO_SENT",
                      "Tax Invoice": "INVOICE_SENT",
                      "Clarification": "AWAITING_CLARIFICATION"}.get(doc_type, "UNKNOWN")

        prev = get_client_status(target)
        prev_status = prev.get("status") if prev else None

        rewrite = doc_type in ("Quotation", "Clarification") and not state.get("retain_items", False)
        await asyncio.to_thread(update_client_status, target, state.get("company_name") or "",
                                state.get("requested_items") or [], new_status, not rewrite)

        if doc_type == "Tax Invoice" and prev_status not in ("INVOICE_SENT", "COMPLETED_AND_RECORDED"):
            ref = os.path.basename(state.get("generated_doc_path") or "") or f"INV-{uuid.uuid4().hex[:6]}"
            await asyncio.to_thread(record_sale, target, state.get("company_name") or "", ref)

    return {"error_message": None}


# =============================================================================
# 11. GRAPH
# =============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_requirements)
workflow.add_node("check_stock", check_stock)
workflow.add_node("generate_quote", generate_quote)
workflow.add_node("generate_lpo", generate_lpo)
workflow.add_node("generate_invoice", generate_invoice)
workflow.add_node("ask_owner_discount", ask_owner_discount)
workflow.add_node("apply_discount_and_requote", apply_discount_and_requote)
workflow.add_node("answer_owner", answer_owner)
workflow.add_node("human_handoff", human_handoff)
workflow.add_node("ignore_email", ignore_email)
workflow.add_node("dispatch", dispatch_and_update)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_workflow, {
     "check_stock": "check_stock",
    "generate_quote": "generate_quote", "generate_lpo": "generate_lpo",
    "generate_invoice": "generate_invoice", "ask_owner_discount": "ask_owner_discount",
    "apply_discount_and_requote": "apply_discount_and_requote",
    "answer_owner": "answer_owner", "human_handoff": "human_handoff",
    "dispatch": "dispatch", "ignore_email": "ignore_email"})

workflow.add_conditional_edges("check_stock", route_workflow, {
     "check_stock": "generate_quote", 
     "ask_owner_discount": "ask_owner_discount"
})
workflow.add_edge("apply_discount_and_requote", "generate_quote")
for node in ["generate_quote", "generate_lpo", "generate_invoice",
             "ask_owner_discount", "answer_owner", "human_handoff"]:
    workflow.add_edge(node, "dispatch")
workflow.add_edge("ignore_email", END)
workflow.add_edge("dispatch", END)
app = workflow.compile()


# =============================================================================
# 12. DASHBOARD
# =============================================================================
def dashboard_data() -> dict:
    with db_conn() as conn:
        daily = {r["d"]: dict(r) for r in conn.execute("""
            SELECT date(sold_at) AS d, SUM(net_revenue) AS rev,
                   SUM(gross_profit) AS pr, SUM(quantity) AS q
            FROM sales WHERE date(sold_at) >= date('now','localtime','-29 days')
            GROUP BY d ORDER BY d""").fetchall()}

    today = datetime.now().date()
    labels, rev_series, profit_series, qty_series = [], [], [], []
    for i in range(29, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        row = daily.get(d)
        labels.append(d[5:].replace("-", "/"))
        rev_series.append(round(row["rev"], 2) if row else 0)
        profit_series.append(round(row["pr"], 2) if row else 0)
        qty_series.append(row["q"] if row else 0)

    by_model = sales_rows("all")
    month = sales_rows("month")
    today_rows = sales_rows("today")
    stock = stock_rows()
    pipe = pipeline_rows()

    def tot(rows, k):
        return sum(r[k] for r in rows)

    return {
        "labels": labels, "rev_series": rev_series,
        "profit_series": profit_series, "qty_series": qty_series,
        "by_model": by_model, "month": month, "today": today_rows,
        "stock": stock, "pipeline": pipe,
        "kpi": {
            "today_units": tot(today_rows, "units"), "today_rev": tot(today_rows, "revenue"),
            "month_units": tot(month, "units"), "month_rev": tot(month, "revenue"),
            "month_profit": tot(month, "profit"),
            "all_units": tot(by_model, "units"), "all_rev": tot(by_model, "revenue"),
            "all_profit": tot(by_model, "profit"),
            "stock_total": sum(s["stock"] for s in stock),
            "low_count": sum(1 for s in stock if s["low"]),
            "pipe_count": len(pipe), "pipe_value": sum(p["value"] for p in pipe),
        },
    }


DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Aerotech Live Dashboard</title>
<script src="[https://cdn.jsdelivr.net/npm/chart.js](https://cdn.jsdelivr.net/npm/chart.js)"></script>
<style>
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#eef2f6;margin:0;padding:12px;color:#2f4256;font-size:15px}
  .wrap{max-width:1000px;margin:0 auto}
  header{background:linear-gradient(135deg,#1b4f72,#2980b9);color:#fff;border-radius:14px;
         padding:18px 16px;margin-bottom:12px}
  header h1{margin:0;font-size:19px;letter-spacing:.3px}
  header p{margin:4px 0 0;font-size:12px;opacity:.85}
  .kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:12px}
  @media(min-width:640px){.kpis{grid-template-columns:repeat(4,1fr)}}
  .kpi{background:#fff;border-radius:12px;padding:13px 12px;box-shadow:0 1px 3px rgba(20,50,80,.08)}
  .kpi .l{font-size:10.5px;color:#8496a6;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
  .kpi .v{font-size:20px;font-weight:700;color:#1b4f72;margin-top:3px;line-height:1.15}
  .kpi .s{font-size:11px;color:#8496a6;margin-top:2px}
  .kpi.warn .v{color:#c0392b}
  .tabs{display:flex;gap:6px;margin-bottom:12px;overflow-x:auto;padding-bottom:2px}
  .tab{flex:0 0 auto;padding:9px 15px;border-radius:999px;background:#fff;border:none;
       font-size:13.5px;font-weight:600;color:#5a6b7b;cursor:pointer;font-family:inherit}
  .tab.on{background:#1b4f72;color:#fff}
  .panel{display:none}.panel.on{display:block}
  .card{background:#fff;border-radius:14px;padding:14px;margin-bottom:12px;
        box-shadow:0 1px 3px rgba(20,50,80,.08)}
  .card h2{margin:0 0 10px;font-size:14px;color:#5a6b7b;text-transform:uppercase;letter-spacing:.6px}
  .chart{position:relative;height:230px}
  @media(min-width:640px){.chart{height:300px}}
  table{width:100%;border-collapse:collapse}
  th{font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:#8496a6;
     text-align:left;padding:8px 6px;border-bottom:2px solid #e3eaf0;white-space:nowrap;
     cursor:pointer;user-select:none}
  th.r,td.r{text-align:right}
  td{padding:10px 6px;border-bottom:1px solid #f0f4f7;font-size:14px}
  tr:last-child td{border-bottom:none}
  .nm{font-weight:600}
  .bad{color:#c0392b;font-weight:700}.mid{color:#e67e22;font-weight:600}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
        background:#eef4f9;color:#1b4f72;font-weight:600}
  .empty{text-align:center;color:#8496a6;padding:24px 0;font-size:14px}
  footer{text-align:center;font-size:11px;color:#9aa8b5;padding:8px 0 16px}
</style></head><body><div class="wrap">

<header><h1>Aerotech Drones</h1><p>Live ERP dashboard &middot; __STAMP__</p></header>

<div class="kpis">
  <div class="kpi"><div class="l">Sold today</div><div class="v">__K_TODAY_U__</div>
       <div class="s">__K_TODAY_R__</div></div>
  <div class="kpi"><div class="l">This month</div><div class="v">__K_MONTH_U__</div>
       <div class="s">__K_MONTH_R__</div></div>
  <div class="kpi"><div class="l">All-time revenue</div><div class="v">__K_ALL_R__</div>
       <div class="s">__K_ALL_U__ drones</div></div>
  <div class="kpi __LOWCLS__"><div class="l">Low stock</div><div class="v">__K_LOW__</div>
       <div class="s">__K_STOCK__ units on hand</div></div>
</div>

<div class="tabs">
  <button class="tab on" data-p="p1">Overview</button>
  <button class="tab" data-p="p2">Models</button>
  <button class="tab" data-p="p3">Stock</button>
  <button class="tab" data-p="p4">Pipeline</button>
</div>

<div class="panel on" id="p1">
  <div class="card"><h2>Revenue &amp; profit &mdash; last 30 days</h2>
    <div class="chart"><canvas id="trend"></canvas></div></div>
  <div class="card"><h2>Units sold per day</h2>
    <div class="chart"><canvas id="units"></canvas></div></div>
</div>

<div class="panel" id="p2">
  <div class="card"><h2>Units by model &mdash; all time</h2>
    <div class="chart"><canvas id="models"></canvas></div></div>
  <div class="card"><h2>Sales by model</h2>__T_MODELS__</div>
  <div class="card"><h2>This month by model</h2>__T_MONTH__</div>
</div>

<div class="panel" id="p3">
  <div class="card"><h2>Stock vs reorder level</h2>
    <div class="chart"><canvas id="stock"></canvas></div></div>
  <div class="card"><h2>Stock detail</h2>__T_STOCK__</div>
</div>

<div class="panel" id="p4">
  <div class="card"><h2>Pipeline Value Distribution</h2>
    <div class="chart"><canvas id="pipe_chart"></canvas></div></div>
  <div class="card"><h2>Open deals &mdash; __K_PIPE_N__ worth __K_PIPE_V__</h2>__T_PIPE__</div>
</div>

<div class="card" style="background:linear-gradient(135deg, #1b4f72, #2980b9); color:#fff; margin-top:24px;">
  <h2 style="color:#fff; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:8px;">Executive Summary</h2>
  <p style="font-size:15px; line-height:1.6; margin-bottom:0; opacity: 0.95;">__SUMMARY_TEXT__</p>
</div>

<footer>Figures read from the immutable sales ledger.</footer>
</div>
<script>
document.querySelectorAll('.tab').forEach(function(b){
  b.onclick=function(){
    document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('on')});
    document.querySelectorAll('.panel').forEach(function(x){x.classList.remove('on')});
    b.classList.add('on');
    document.getElementById(b.dataset.p).classList.add('on');
    window.dispatchEvent(new Event('resize'));
  };
});

document.querySelectorAll('table').forEach(function(t){
  t.querySelectorAll('th').forEach(function(h,i){
    h.onclick=function(){
      var body=t.tBodies[0], rows=Array.prototype.slice.call(body.rows);
      var dir=h.dataset.d==='1'?-1:1; h.dataset.d=dir===1?'1':'0';
      rows.sort(function(a,b){
        var x=a.cells[i].dataset.v!==undefined?parseFloat(a.cells[i].dataset.v):a.cells[i].innerText;
        var y=b.cells[i].dataset.v!==undefined?parseFloat(b.cells[i].dataset.v):b.cells[i].innerText;
        if(typeof x==='number'&&!isNaN(x))return (x-y)*dir;
        return String(x).localeCompare(String(y))*dir;
      });
      rows.forEach(function(r){body.appendChild(r)});
    };
  });
});

var CUR='__CURRENCY__';
var fmt={responsive:true,maintainAspectRatio:false,
         plugins:{legend:{labels:{boxWidth:12,font:{size:11}}}},
         scales:{x:{ticks:{font:{size:9},maxRotation:0,autoSkip:true,maxTicksLimit:8}},
                 y:{beginAtZero:true,ticks:{font:{size:10}}}}};

new Chart(document.getElementById('trend'),{type:'line',
  data:{labels:__LABELS__,datasets:[
    {label:'Revenue',data:__REV__,borderColor:'#2980b9',backgroundColor:'rgba(41,128,185,.12)',
     fill:true,tension:.3,borderWidth:2,pointRadius:0},
    {label:'Gross profit',data:__PROFIT__,borderColor:'#27ae60',
     backgroundColor:'rgba(39,174,96,.10)',fill:true,tension:.3,borderWidth:2,pointRadius:0}
  ]},options:fmt});

new Chart(document.getElementById('units'),{type:'bar',
  data:{labels:__LABELS__,datasets:[{label:'Units',data:__QTY__,backgroundColor:'#5dade2'}]},
  options:fmt});

new Chart(document.getElementById('models'),{type:'bar',
  data:{labels:__M_LABELS__,datasets:[{label:'Units sold',data:__M_UNITS__,
        backgroundColor:'#1b4f72'}]},
  options:Object.assign({},fmt,{indexAxis:'y',
    scales:{x:{beginAtZero:true,ticks:{font:{size:10}}},y:{ticks:{font:{size:10}}}}})});

new Chart(document.getElementById('stock'),{type:'bar',
  data:{labels:__S_LABELS__,datasets:[
    {label:'In stock',data:__S_STOCK__,backgroundColor:'#5dade2'},
    {label:'Reorder level',data:__S_LEVEL__,backgroundColor:'#e74c3c'}]},
  options:Object.assign({},fmt,{indexAxis:'y',
    scales:{x:{beginAtZero:true,ticks:{font:{size:10}}},y:{ticks:{font:{size:10}}}}})});

new Chart(document.getElementById('pipe_chart'),{
  type:'doughnut',
  data:{
    labels:__P_LABELS__,
    datasets:[{
      label:'Deal Value',
      data:__P_VALUE__,
      backgroundColor:['#1b4f72','#2980b9','#27ae60','#e67e22','#8e44ad','#e74c3c','#34495e']
    }]
  },
  options: Object.assign({}, fmt, {
    plugins: { legend: { position: 'right', labels: { boxWidth: 12, font: { size: 11 } } } },
    scales: { x: { display: false }, y: { display: false } }
  })
});
</script></body></html>"""


def _dash_table(headers, rows, right_cols=()):
    if not rows:
        return "<div class='empty'>Nothing to show yet.</div>"
    th = "".join(f"<th class='{'r' if i in right_cols else ''}'>{html.escape(str(h))}</th>"
                 for i, h in enumerate(headers))
    body = ""
    for row in rows:
        tds = ""
        for i, cell in enumerate(row):
            if isinstance(cell, tuple):
                disp, sortval = cell
                tds += (f"<td class='{'r' if i in right_cols else ''}' "
                        f"data-v='{sortval}'>{disp}</td>")
            else:
                tds += f"<td class='{'r' if i in right_cols else ''}'>{cell}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def build_dashboard_html() -> str:
    d = dashboard_data()
    k = d["kpi"]

    model_tbl = _dash_table(
        ["Model", "Units", "Revenue", "Gross profit", "Buyers"],
        [[f"<span class='nm'>{html.escape(r['product_name'])}</span>",
          (str(r["units"]), r["units"]), (money(r["revenue"]), r["revenue"]),
          (money(r["profit"]), r["profit"]), (str(r["customers"]), r["customers"])]
         for r in d["by_model"]], right_cols=(1, 2, 3, 4))

    month_tbl = _dash_table(
        ["Model", "Units", "Revenue", "Gross profit"],
        [[f"<span class='nm'>{html.escape(r['product_name'])}</span>",
          (str(r["units"]), r["units"]), (money(r["revenue"]), r["revenue"]),
          (money(r["profit"]), r["profit"])]
         for r in d["month"]], right_cols=(1, 2, 3))

    stock_rows_html = []
    for s in d["stock"]:
        cls = "bad" if s["urgent"] else ("mid" if s["low"] else "")
        left = f"{s['days_left']:g} d" if s["days_left"] is not None else "&mdash;"
        stock_rows_html.append([
            f"<span class='nm'>{html.escape(s['product'])}</span>",
            (f"<span class='{cls}'>{s['stock']}</span>", s["stock"]),
            (str(s["reorder_level"]), s["reorder_level"]),
            (f"{s['velocity']:g}/d", s["velocity"]),
            (f"<span class='{cls}'>{left}</span>",
             s["days_left"] if s["days_left"] is not None else 99999),
            (str(s["sold"]), s["sold"])])
    stock_tbl = _dash_table(
        ["Model", "In stock", "Reorder at", "Velocity", "Runs out", "Sold"],
        stock_rows_html, right_cols=(1, 2, 3, 4, 5))

    pipe_tbl = _dash_table(
        ["Client", "Stage", "Units", "Value"],
        [[f"<span class='nm'>{html.escape(p['client_name'] or p['email'])}</span>",
          f"<span class='pill'>{p['status'].replace('_', ' ').title()}</span>",
          (str(p["units"]), p["units"]), (money(p["value"]), p["value"])]
         for p in d["pipeline"]], right_cols=(2, 3))
         
    summary_text = (
        f"<strong>Performance:</strong> Total all-time revenue stands at {money(k['all_rev'])} "
        f"across {k['all_units']} units sold. This month, you have moved {k['month_units']} units "
        f"generating {money(k['month_rev'])} in revenue.<br><br>"
        "<strong>Inventory:</strong> There are " + str(k['stock_total']) + " units currently on hand. " +
        (f"<span style=\"color:#f1c40f; font-weight:bold;\">Attention is required for {k['low_count']} model(s) running low on stock.</span>" if k['low_count'] > 0 else "All inventory levels are healthy.") + "<br><br>"
        f"<strong>Pipeline:</strong> You have {k['pipe_count']} active open deals worth a potential {money(k['pipe_value'])}."
    )

    subs = {
        "__STAMP__": datetime.now().strftime("%d %b %Y, %H:%M"),
        "__CURRENCY__": CURRENCY,
        "__K_TODAY_U__": str(k["today_units"]), "__K_TODAY_R__": money(k["today_rev"]),
        "__K_MONTH_U__": str(k["month_units"]), "__K_MONTH_R__": money(k["month_rev"]),
        "__K_ALL_R__": money(k["all_rev"]), "__K_ALL_U__": str(k["all_units"]),
        "__K_LOW__": str(k["low_count"]), "__K_STOCK__": str(k["stock_total"]),
        "__LOWCLS__": "warn" if k["low_count"] else "",
        "__K_PIPE_N__": f"{k['pipe_count']} deals", "__K_PIPE_V__": money(k["pipe_value"]),
        "__LABELS__": json.dumps(d["labels"]), "__REV__": json.dumps(d["rev_series"]),
        "__PROFIT__": json.dumps(d["profit_series"]), "__QTY__": json.dumps(d["qty_series"]),
        "__M_LABELS__": json.dumps([r["product_name"] for r in d["by_model"]]),
        "__M_UNITS__": json.dumps([r["units"] for r in d["by_model"]]),
        "__S_LABELS__": json.dumps([s["product"] for s in d["stock"]]),
        "__S_STOCK__": json.dumps([s["stock"] for s in d["stock"]]),
        "__S_LEVEL__": json.dumps([s["reorder_level"] for s in d["stock"]]),
        "__P_LABELS__": json.dumps([p["client_name"] or p["email"] for p in d["pipeline"]]),
        "__P_VALUE__": json.dumps([p["value"] for p in d["pipeline"]]),
        "__SUMMARY_TEXT__": summary_text,
        "__T_MODELS__": model_tbl, "__T_MONTH__": month_tbl,
        "__T_STOCK__": stock_tbl, "__T_PIPE__": pipe_tbl,
    }
    out = DASHBOARD_TEMPLATE
    for token, value in subs.items():
        out = out.replace(token, value)
    return out


def send_owner_report_sync():
    page = build_dashboard_html()
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(page)

    summary = build_owner_report("sales this month")

    msg = MIMEMultipart("mixed")
    msg["From"], msg["To"] = EMAIL_USER, EMAIL_USER
    msg["X-Drone-Bot"] = "true"
    msg["Auto-Submitted"] = "auto-replied"
    msg["Subject"] = "Aerotech - interactive dashboard"

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(summary["plain"] +
                        "\n\nOpen the attached dashboard.html for charts, "
                        "stock cover and the full pipeline.", "plain"))
    alt.attach(MIMEText(summary["html"], "html"))
    msg.attach(alt)

    part = MIMEApplication(page.encode("utf-8"), Name="dashboard.html")
    part["Content-Disposition"] = 'attachment; filename="dashboard.html"'
    msg.attach(part)

    send_email_sync(msg)
    logging.info("Dashboard emailed to owner.")


# =============================================================================
# 13. IMAP INGESTION
# =============================================================================
def extract_message_body(msg) -> str:
    if msg.is_multipart():
        html_fallback = ""
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if "attachment" in str(part.get("Content-Disposition", "")):
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
            if part.get_content_type() == "text/plain":
                return text
            if part.get_content_type() == "text/html" and not html_fallback:
                html_fallback = re.sub(r"<[^>]+>", " ", text)
        return html_fallback.strip()

    payload = msg.get_payload(decode=True) or b""
    text = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
    return re.sub(r"<[^>]+>", " ", text).strip() if msg.get_content_type() == "text/html" else text


def fetch_one(mail, uid: int) -> Optional[dict]:
    res, msg_data = mail.uid("fetch", str(uid), "(RFC822)")
    if res != "OK" or not msg_data:
        return None
    raw = next((p[1] for p in msg_data if isinstance(p, tuple)), None)
    if raw is None:
        return None

    msg = email.message_from_bytes(raw)
    mail.uid("store", str(uid), "+FLAGS", "(\\Seen)")

    subject = decode_header_value(msg.get("subject"))         
    message_id = msg.get("Message-ID") or f"uid-{uid}"
    display_name, sender = email.utils.parseaddr(msg.get("from", ""))
    display_name = decode_header_value(display_name)
    sender = (sender or "").strip()

    if msg.get("X-Drone-Bot") == "true" or msg.get("Auto-Submitted") == "auto-replied":
        return None
    if already_processed(message_id):
        logging.info(f"Duplicate Message-ID skipped ({sender}).")
        return None
    if not sender:
        return None

    sub_lower = subject.lower()
    if sender.lower() == EMAIL_USER.lower() and "send" in sub_lower and "report" in sub_lower:
        mark_processed(message_id)
        send_owner_report_sync()
        record_trace(DB_FILE, sender, subject, extract_message_body(msg),
                     None, "owner_report", "Dashboard", "", [], 0)
        return None
    if any(d in sender.lower() for d in IGNORE_DOMAINS):
        return None

    mark_processed(message_id)
    return {"email_id": str(uid), "message_id": message_id, "sender_email": sender,
            "display_name": display_name, "email_subject": subject,
            "email_body": extract_message_body(msg)}


def fetch_unread_emails() -> List[dict]:
    out, mail = [], None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select(MAILBOX)

        last = get_last_uid()

        if last == 0:
            status, messages = mail.uid("search", None, "UNSEEN")
            if status != "OK" or not messages or not messages[0]:
                return []
            uids = sorted(int(u) for u in messages[0].split())
            set_last_uid(uids[-1])
            logging.info(f"First run: watermark set to UID {uids[-1]}; "
                         f"{len(uids)} pre-existing unread message(s) skipped.")
            return []

        status, messages = mail.uid("search", None, f"UID {last + 1}:*", "UNSEEN")
        if status != "OK" or not messages or not messages[0]:
            return []

        uids = sorted(int(u) for u in messages[0].split())
        pending = [u for u in uids if u > last]
        if len(pending) > MAX_EMAILS_PER_CYCLE:
            logging.info(f"{len(pending)} new messages; taking {MAX_EMAILS_PER_CYCLE} "
                         f"this cycle, remainder next cycle.")

        for uid in pending[:MAX_EMAILS_PER_CYCLE]:
            try:
                data = fetch_one(mail, uid)
            except Exception as e:
                logging.error(f"UID {uid} unparseable ({e}); skipping.")
                set_last_uid(uid)
                continue
            set_last_uid(uid)      
            if data:
                out.append(data)

    except Exception as e:
        logging.error(f"IMAP fetch failed: {e}")
    finally:
        if mail:
            try:
                mail.logout()
            except Exception:
                pass
    return out


# =============================================================================
# 14. WORKERS
# =============================================================================
async def email_poller(queue: asyncio.Queue):
    while True:
        try:
            for mail in await asyncio.to_thread(fetch_unread_emails):
                await queue.put(mail)
        except Exception as e:
            logging.error(f"Poller error: {e}")
        await asyncio.to_thread(heartbeat, DB_FILE)
        await asyncio.sleep(POLL_SECONDS)


async def agent_worker(queue: asyncio.Queue):
    while True:
        mail_data = await queue.get()
        _t0 = datetime.now()
        _prev = get_client_status(mail_data.get("sender_email", ""))
        _status_before = _prev.get("status") if _prev else None
        LAST_LLM_RAW["v"] = ""
        _final = None
        try:
            _final = await app.ainvoke({
                "email_id": mail_data.get("email_id", ""),
                "message_id": mail_data.get("message_id", ""),
                "sender_email": mail_data.get("sender_email", ""),
                "display_name": mail_data.get("display_name", ""),
                "email_subject": mail_data.get("email_subject", ""),
                "email_body": mail_data.get("email_body", ""),
                "intent": "unrelated",
                "company_name": mail_data.get("display_name", ""),
                "requested_items": [], "feedback_reason": "",
                "target_client_email": None, "generated_doc_path": None,
                "doc_type_sent": None, "reply_message": None,
                "reply_html": None, "reply_subject": None,
                "retain_items": False, "handoff_note": None, "error_message": None})
            
            record_trace(
                DB_FILE, mail_data.get("sender_email", ""),
                mail_data.get("email_subject", ""), mail_data.get("email_body", ""),
                _status_before, (_final or {}).get("intent"),
                (_final or {}).get("doc_type_sent"), LAST_LLM_RAW["v"],
                (_final or {}).get("requested_items"),
                (datetime.now() - _t0).total_seconds() * 1000)
        except Exception as e:
            logging.error(f"Agent worker error on {mail_data.get('sender_email')}: {e}")
        finally:
            queue.task_done()


def apply_pending_stock():
    with db_conn() as conn:
        c = conn.cursor()
        pending = c.execute("SELECT id, order_email, product_name, quantity "
                            "FROM sales WHERE stock_applied=0").fetchall()
        touched = set()
        for row in pending:
            inv = c.execute("SELECT total_stock FROM inventory "
                            "WHERE product_name=? COLLATE NOCASE",
                            (row["product_name"],)).fetchone()
            if inv is None:
                logging.warning(f"LEDGER line {row['id']}: '{row['product_name']}' not in "
                                f"inventory. Not deducted.")
                continue
            if inv["total_stock"] < row["quantity"]:
                logging.warning(f"OVERSELL: {row['quantity']}x {row['product_name']} invoiced "
                                f"but only {inv['total_stock']} in stock.")
            c.execute("""UPDATE inventory SET total_stock = MAX(total_stock - ?, 0),
                                              total_sales = total_sales + ?
                         WHERE product_name=? COLLATE NOCASE""",
                      (row["quantity"], row["quantity"], row["product_name"]))
            c.execute("UPDATE sales SET stock_applied=1 WHERE id=?", (row["id"],))
            touched.add(row["order_email"])
            logging.info(f"STOCK -{row['quantity']} {row['product_name']}")
        for e in touched:
            c.execute("UPDATE orders SET status='COMPLETED_AND_RECORDED' "
                      "WHERE email=? AND status='INVOICE_SENT'", (e,))


async def inventory_manager_worker():
    logging.info("Inventory worker online.")
    while True:
        try:
            await asyncio.to_thread(apply_pending_stock)
            await asyncio.to_thread(check_low_stock)
        except Exception as e:
            logging.error(f"Inventory worker error: {e}")
        await asyncio.sleep(INVENTORY_SWEEP_SECONDS)


async def main():
    q = asyncio.Queue()
    logging.info("Aerotech ERP v5.2 starting...")
    logging.info(f"Database: {DB_FILE} | Mailbox: {MAILBOX} | Poll: {POLL_SECONDS}s")
    await asyncio.gather(email_poller(q), agent_worker(q), inventory_manager_worker())


if __name__ == "__main__":
    asyncio.run(main())
