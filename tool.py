# ------------------------------------------------------------
# Forensic Audio Classifier Tool
# Author: Arnab Das
# Year: 2025
# Version: 0.9
# ------------------------------------------------------------


import os
import re
import csv
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSequenceClassification
import sys

# Optional LM decoder
try:
    from pyctcdecode import build_ctcdecoder
    HAVE_PYCTC = True
except Exception:
    HAVE_PYCTC = False

# ---------------- CONFIG (GitHub Ready) ----------------
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load external config.json
with open(os.path.join(BASE_DIR, "config.json"), "r", encoding="utf-8") as jf:
    CONFIG = json.load(jf)

# Pre-Trained Model from HuggingFace
ACOUSTIC_MODEL_MAP = {
    "Bangla_Acoustic_Model": "sazzadul/Shrutimala_Bangla_ASR"
}

# Resolve acoustic model repo
ACOUSTIC_MODEL_REPO = ACOUSTIC_MODEL_MAP.get(CONFIG["MODEL_PATH"], CONFIG["MODEL_PATH"])

# ---------------- GUI OVERRIDE (FOR FUTURE GUI INTEGRATION) ----------------
if len(sys.argv) > 1:
    CONFIG["AUDIO_FOLDER"] = sys.argv[1]
    print(f"[GUI] Overriding AUDIO_FOLDER -> {CONFIG['AUDIO_FOLDER']}")

# ---------------- Phonetic & vowel normalization groups ----------------
PHONETIC_GROUPS = [
    ("শষস", "শ"),
    ("ড়ঢ়র", "র"),
    ("নণ", "ন"),
    ("বভ", "ব"),
    ("জয", "জ"),
    ("চছশ", "চ"),
    ("তট", "ত"),
    ("দড", "দ"),
    ("ল঳", "ল"),
    ("ঙগঘ", "গ"),
    ("কখ", "ক"),
    ("ফপ", "প"),
    ("রঋ", "র"),
    ("ওঅঔ", "ও"),
    ("ইঈ", "ই"),
    ("উঊ", "উ"),
    ("এঐ", "এ"),
    ("অআ", "আ"),
    ("িী", "ি"),
    ("ুূ", "ু"),
    ("োৌ", "ো"),
    ("েৈ", "ে"),
    ("অা", "া"),
]

SEV_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Ensure output dirs exist
os.makedirs(CONFIG["FLAGGED_DIR"], exist_ok=True)
os.makedirs(CONFIG["SAFE_DIR"], exist_ok=True)
os.makedirs(CONFIG["REVIEW_DIR"], exist_ok=True)

def log_event(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\\n"
    with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
        f.write(line)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")

def phonetic_normalize(word: str) -> str:
    if not isinstance(word, str):
        return word
    out = word
    for group, rep in PHONETIC_GROUPS:
        for ch in group:
            out = out.replace(ch, rep)
    return out

COMMON_SUFFIXES = [
    "ছি", "ছেন", "ছিল", "ছিলাম", "লাম", "লে", "লা", "লি", "রা", "রো", "ছিলে",
    "তে", "ান", "েন", "ছি", "েছি", "ছে", "ছিলি"
]

def light_stem(word: str) -> str:
    if not isinstance(word, str):
        return word
    for suf in COMMON_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[:-len(suf)]
    return word

def tokenize_bn(text: str):
    return re.findall(r"[\u0980-\u09FF]+", str(text))

def bengali_overlap(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def load_keywords(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and not f.startswith("context_tokens")]
    dfs = []
    for f in files:
        try:
            path = os.path.join(folder, f)
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.columns = [normalize_col(c) for c in df.columns]
            if {"keyword", "severity", "category"}.issubset(set(df.columns)):
                dfs.append(df[["keyword", "severity", "category"]])
        except Exception as e:
            log_event(f"Failed to load keyword file {f}: {e}")
    if not dfs:
        return pd.DataFrame(columns=["keyword", "severity", "category"])
    out = pd.concat(dfs, ignore_index=True)
    out["keyword"] = out["keyword"].astype(str).str.strip()
    out["severity"] = out["severity"].astype(str).str.lower().str.strip()
    out["category"] = out["category"].astype(str).str.strip()
    return out

def load_context_tokens(path: str):
    try:
        df = pd.read_csv(path, header=None, names=["token"], encoding="utf-8-sig")
        tokens = df["token"].dropna().astype(str).str.strip().tolist()
        return [t for t in tokens if t]
    except Exception:
        return []

print("Loading ASR processor & Acoustic Model...")
processor = AutoProcessor.from_pretrained(ACOUSTIC_MODEL_REPO)
am_model = AutoModelForCTC.from_pretrained(ACOUSTIC_MODEL_REPO).to(device).eval()

decoder = None
use_lm = False
if HAVE_PYCTC and os.path.exists(CONFIG["LM_ARPA"]) and os.path.exists(CONFIG["VOCAB_PATH"]):
    try:
        print("Building CTC decoder via pyctcdecode + KenLM...")
        vocab_dict = processor.tokenizer.get_vocab()
        maxid = max(vocab_dict.values())
        id_to_token = [None] * (maxid + 1)
        for tok, idx in vocab_dict.items():
            id_to_token[idx] = tok
        id_to_token = [t for t in id_to_token if t is not None]
        unigrams = [w.strip() for w in open(CONFIG["VOCAB_PATH"], encoding="utf-8") if w.strip()]
        decoder = build_ctcdecoder(labels=id_to_token, kenlm_model_path=CONFIG["LM_ARPA"], unigrams=unigrams)
        use_lm = True
        print("✅ Decoder built — AM+LM enabled.")
    except Exception as e:
        log_event(f"Decoder build failed: {e}")
        print("⚠️ Decoder build failed — falling back to greedy decoding. Error:", e)
else:
    print("pyctcdecode or LM files unavailable — using greedy decode.")

print("Loading classifier model...")
tokenizer_clf = AutoTokenizer.from_pretrained(CONFIG["CLASSIFIER_PATH"])
clf_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["CLASSIFIER_PATH"]).to(device).eval()


def detect_pos_idx(config):
    id2label = getattr(config, "id2label", None)
    if not id2label:
        return 1
    for k, v in id2label.items():
        sval = str(v).lower()
        if any(tok in sval for tok in ("hate", "abuse", "abusive", "offensive", "violent", "threat", "crime", "weapon", "gun", "yes", "1", "positive")):
            try:
                return int(k)
            except:
                return int(k)
    if len(id2label) == 2:
        return 1
    return 1

pos_idx = detect_pos_idx(clf_model.config)
print("Assuming positive class index (pos_idx) =", pos_idx)

def detect_keywords_for_display(text: str, keywords_df: pd.DataFrame, display_thr: int = 80):
    words = tokenize_bn(text)
    if not words or keywords_df.empty:
        return []
    matches = []
    for _, row in keywords_df.iterrows():
        kw_orig = str(row.keyword).strip()
        sev = str(row.severity).lower().strip()
        cat = str(row.category).strip()
        kw_norm = phonetic_normalize(light_stem(kw_orig))
        klen = len(kw_norm)
        for w in words:
            w_norm = phonetic_normalize(light_stem(w))
            if abs(klen - len(w_norm)) > 2:
                continue
            if klen <= 5:
                if kw_norm == w_norm:
                    matches.append({"keyword": kw_orig, "matched_word": w, "severity": sev, "category": cat, "score": 100})
                continue
            score1 = fuzz.ratio(kw_norm, w_norm)
            score2 = fuzz.partial_ratio(kw_norm, w_norm)
            score = max(score1, score2)
            if score < display_thr:
                continue
            overlap = bengali_overlap(kw_norm, w_norm)
            if overlap < 0.9:
                continue
            matches.append({"keyword": kw_orig, "matched_word": w, "severity": sev, "category": cat, "score": round(float(score), 1)})
    uniq = []
    seen = set()
    for m in sorted(matches, key=lambda x: x["score"], reverse=True):
        key = (m["keyword"], m["matched_word"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq

def high_severity_keyword_flag(matches, keyflag_thr: int = 90) -> bool:
    for m in matches:
        if m["severity"] in ("critical", "high") and m["score"] >= keyflag_thr:
            return True
    return False

def highest_severity_score(matches) -> int:
    max_rank = 0
    for m in matches:
        r = SEV_RANK.get(m["severity"].lower(), 0)
        if r > max_rank:
            max_rank = r
    return max_rank

keywords_df = load_keywords(CONFIG["KEYWORDS_FOLDER"])
context_tokens = load_context_tokens(CONFIG["CONTEXT_FILE"])
print(f"🔑 Loaded {len(keywords_df)} keywords. 💬 Loaded {len(context_tokens)} context tokens.")

def context_present(text: str) -> bool:
    t = str(text).lower()
    for ct in context_tokens:
        if ct and ct.lower() in t:
            return True
    return False

def load_audio(path: str, sr: int = 16000):
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    denom = float(waveform.abs().max().item()) if waveform.abs().max().item() != 0 else 1.0
    waveform = waveform / denom
    return waveform, sr

def trim_silence(wave, th: float = 0.01):
    mask = wave.abs() > th
    if not mask.any():
        return wave
    idx = mask.nonzero(as_tuple=True)[0]
    return wave[idx[0]: idx[-1] + 1]

audio_files = [
    os.path.join(CONFIG["AUDIO_FOLDER"], f)
    for f in os.listdir(CONFIG["AUDIO_FOLDER"])
    if f.lower().endswith((".wav", ".mp3", ".opus"))
]

print(f"📂 Using audio folder: {CONFIG['AUDIO_FOLDER']}")

results = []

# timestamped filenames
ts = datetime.now().strftime("%Y-%m-%d_%H%M")
csv_path = os.path.join(CONFIG["OUTPUT_DIR"], f"crime_detection_report_{ts}.csv")
html_path = os.path.join(CONFIG["OUTPUT_DIR"], f"crime_detection_report_{ts}.html")
log_path = os.path.join(CONFIG["OUTPUT_DIR"], f"logs_{ts}.txt")

# start logging header
with open(log_path, "a", encoding="utf-8") as lf:
    lf.write(f"Run started: {datetime.now().isoformat()}\\n")

print("\\n🔊 Starting analysis — running through audio files...\\n")
for path in audio_files:
    filename = os.path.basename(path)
    try:
        waveform, sr = load_audio(path, CONFIG["TARGET_SR"])
    except Exception as e:
        log_event(f"ERROR loading {filename}: {e}")
        print(f"❌ Failed to load {filename}: {e}")
        continue

    waveform = trim_silence(waveform)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_tensor = inputs.get("input_values") or inputs.get("input_features")
    if input_tensor is None:
        log_event(f"Processor returned no input tensor for {filename}")
        print(f"⚠️ Processor returned no inputs for {filename}")
        continue
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        am_out = am_model(input_tensor)
        logits_am = am_out.logits  # (B, T, V)

    try:
        pred_ids = torch.argmax(logits_am, dim=-1)
        greedy_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    except Exception:
        greedy_text = ""

    transcription = greedy_text
    used_lm = False
    if use_lm and decoder is not None:
        try:
            logits_np = logits_am[0].cpu().numpy()
            probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            lm_text = decoder.decode(probs)
            transcription = lm_text.strip()
            used_lm = True
        except Exception as e:
            transcription = greedy_text or ""
            used_lm = False
            log_event(f"LM decode failed for {filename}: {e}")

    text = transcription.strip() or greedy_text.strip() or ""
    if not text:
        text = ""

    try:
        tok_inputs = tokenizer_clf(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            logits_clf = clf_model(**tok_inputs).logits.squeeze(0) / float(CONFIG["TEMP"])
            probs = F.softmax(logits_clf, dim=-1).cpu()
    except Exception as e:
        log_event(f"Classifier error for {filename}: {e}")
        probs = torch.tensor([1.0, 0.0]) if isinstance(pos_idx, int) and pos_idx == 0 else torch.tensor([0.0, 1.0])

    probs_np = probs.numpy().flatten()
    pos_prob = float(probs_np[pos_idx]) if pos_idx < len(probs_np) else float(probs_np.max())

    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = float((sorted_probs[0] - sorted_probs[1]).item()) if sorted_probs.numel() >= 2 else 1.0

    matches_display = detect_keywords_for_display(text, keywords_df, CONFIG["DISPLAY_THRESHOLD"])
    kw_force = high_severity_keyword_flag(matches_display, CONFIG["KEYFLAG_THRESHOLD"])
    max_kw_rank = highest_severity_score(matches_display)
    ctx = context_present(text)

    final_score = CONFIG["ALPHA"] * pos_prob + CONFIG["BETA"] * (max_kw_rank / 4.0)

    # New hierarchical decision logic as requested by user
    status = "safe"
    flag_source = "none"
    reason = ""

    if not matches_display and flag_source == "none" and pos_prob >= CONFIG["CLASSIFIER_THRESHOLD_FLAG"]:
        # no keywords, flagged by classifier alone — apply final_score threshold rule
        if final_score >= 0.7:
            status = "flagged"; flag_source = "classifier"; reason = "classifier_alone_final>=0.7"
        else:
            status = "review"; flag_source = "classifier"; reason = "classifier_alone_final<0.7"
    elif matches_display and pos_prob >= CONFIG["CLASSIFIER_THRESHOLD_FLAG"]:
        status = "flagged"; flag_source = "classifier+keyword"; reason = "kw_present_and_classifier_flag"
    elif matches_display and pos_prob < CONFIG["CLASSIFIER_THRESHOLD_FLAG"]:
        # keyword present but classifier not convinced — severity decides
        top_sev = max([m.get("severity","").lower() for m in matches_display], default="")
        if top_sev in ("critical", "high"):
            status = "flagged"; flag_source = "keyword_high"; reason = "keyword_high_but_clf_low"
        else:
            status = "review"; flag_source = "keyword_low"; reason = "keyword_low_clf_low"
    elif not matches_display and pos_prob < CONFIG["CLASSIFIER_THRESHOLD_SAFE"]:
        status = "safe"; flag_source = "classifier_safe"; reason = "classifier_safe_no_kw"
    else:
        # fallback to earlier hybrid logic with margin
        if kw_force and ctx:
            status = "review"; flag_source = "keyword_context"; reason = "kw_force+context"
        elif kw_force:
            status = "flagged"; flag_source = "keyword"; reason = "kw_force"
        elif pos_prob >= CONFIG["CLASSIFIER_THRESHOLD_FLAG"]:
            if margin < CONFIG["MARGIN_REVIEW"]:
                status = "review"; flag_source = "margin_review"; reason = "margin_review"
            else:
                status = "flagged"; flag_source = "classifier"; reason = "classifier_confident"
        elif CONFIG["CLASSIFIER_THRESHOLD_SAFE"] < pos_prob < CONFIG["CLASSIFIER_THRESHOLD_FLAG"]:
            status = "review"; flag_source = "classifier_review"; reason = "classifier_mid"
        elif final_score >= CONFIG["FINAL_SCORE_THRESHOLD"]:
            if ctx and max_kw_rank > 0:
                status = "review"; flag_source = "final_context"; reason = "final+context"
            else:
                status = "flagged"; flag_source = "hybrid"; reason = "final_high"
        else:
            status = "safe"; flag_source = "none"; reason = "no_rule_match"

    dst = CONFIG["FLAGGED_DIR"] if status == "flagged" else CONFIG["REVIEW_DIR"] if status == "review" else CONFIG["SAFE_DIR"]
    try:
        shutil.copy2(path, dst)
    except Exception as e:
        log_event(f"File copy error for {filename}: {e}")

    detected_keywords_str = ";".join([f"{m['keyword']}→{m['matched_word']}({m['score']})[{m['severity']}]" for m in matches_display])
    fuzzy_diag = ";".join([f"{m['keyword']}→{m['matched_word']}({m['score']})" for m in matches_display if m['score'] < CONFIG["DISPLAY_THRESHOLD"]][:3])

    # logging entry
    log_event(f"{filename} | status={status} | flag_source={flag_source} | pos_prob={pos_prob:.4f} | final_score={final_score:.4f} | keywords={detected_keywords_str} | reason={reason}")

    emoji = "🚨" if status == "flagged" else ("🧐" if status == "review" else "✅")
    lm_icon = "🧠" if used_lm else "💬"
    print(f"{emoji} {filename} ({lm_icon} LM={used_lm}) => {text}")
    print(f"   └─ pos_prob={pos_prob:.3f} | margin={margin:.3f} | final={final_score:.3f} | flag_source={flag_source} | {status.upper()}")
    if detected_keywords_str:
        print(f"      • keywords: {detected_keywords_str}")
    if fuzzy_diag:
        print(f"      • fuzzy_diagnostics: {fuzzy_diag}")

    results.append({
        "filename": filename,
        "transcription": text,
        "used_lm": used_lm,
        "pos_prob": round(pos_prob, 4),
        "margin": round(margin, 4),
        "max_keyword_severity_rank": int(max_kw_rank),
        "detected_keywords": detected_keywords_str,
        "kw_force_flag": bool(kw_force),
        "context_present": bool(ctx),
        "final_score": round(final_score, 4),
        "status": status,
        "flag_source": flag_source,
        "fuzzy_diagnostics": fuzzy_diag,
        "reason": reason
    })

# ---------------- CSV output ----------------
fieldnames = [
    "filename", "transcription", "used_lm", "pos_prob", "margin",
    "max_keyword_severity_rank", "detected_keywords", "kw_force_flag", "context_present",
    "final_score", "status", "flag_source", "fuzzy_diagnostics", "reason"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

    fout.write("\n# ------------------------------------------------------------\n")
    fout.write("# Forensic Audio Detection Report\n")
    fout.write("# Developed by Arnab Das — Version 0.9 (2025)\n")
    fout.write("# ------------------------------------------------------------\n")

print("\\n📁 CSV report saved to:", csv_path)

# ---------------- HTML report generation ----------------
# Group results by status order: flagged, review, safe
status_order = {"flagged": 0, "review": 1, "safe": 2}
results_sorted = sorted(results, key=lambda r: (status_order.get(r["status"], 3), -r["final_score"]))

# HTML helper: highlight keywords inside transcript
def highlight_transcript(transcript, detected_keywords):
    html = transcript
    # build mapping of keyword -> severity
    kw_map = {}
    for part in detected_keywords.split(";"):
        if not part.strip():
            continue
        try:
            # format: kw→matched(score)[severity]
            base, rest = part.split("→", 1)
            matched, rest2 = rest.split("(", 1)
            score_part = rest2.split(")", 1)[0]
            sev = rest.split("[")[-1].replace("]", "")
            kw_map[matched] = sev.lower()
        except Exception:
            continue
    # sort by length descending to avoid submatches
    for matched, sev in sorted(kw_map.items(), key=lambda x: -len(x[0])):
        color = "red" if sev in ("critical", "high") else "orange"
        # escape for html simple
        safe_matched = matched
        html = re.sub(re.escape(safe_matched), f'<span style="color:{color};font-weight:700;">{safe_matched}</span>', html, flags=re.IGNORECASE)
    return html

html_lines = [
    "<!doctype html>",
    "<html><head><meta charset='utf-8'><title>Forensic Audio Detection Report</title>",
    """
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 25px;
            background: #fafafa;
        }
        .header-box {
            background: #1f2937;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .header-box h1 {
            margin: 0;
            font-size: 26px;
        }
        .header-box p {
            margin: 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .card {
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .flagged { background:#ffefef;border-left:6px solid #ff4d4f; }
        .review  { background:#fff6e6;border-left:6px solid #ffb84d; }
        .safe    { background:#effaf0;border-left:6px solid #4caf50; }
        .meta { font-size:12px;color:#444;margin-bottom:6px; }
        audio { max-width:350px; margin-top:8px; }
        .footer {
            margin-top: 40px;
            text-align:center;
            font-size:12px;
            color:#777;
        }
    </style>
    """,
    "</head><body>",
    "<div class='header-box'>",
    "<h1>Forensic Audio Detection Report</h1>",
    f"<p>Developed by Arnab Das — Version 0.9 (2025)</p>",
    f"<p>Date: {datetime.now().strftime('%d/%m/%Y')}</p>",
    "</div>"
]


for r in results_sorted:
    status = r["status"]
    cls = "flagged" if status == "flagged" else ("review" if status == "review" else "safe")
    color_emoji = "🚨" if status == "flagged" else ("🧐" if status == "review" else "✅")
    audio_rel = os.path.join("Audios", r["filename"])
    transcript_high = highlight_transcript(r["transcription"], r["detected_keywords"])
    html_lines.append(f"<div class='card {cls}'>")
    html_lines.append(f"<h3>{color_emoji} {r['filename']} — {status.upper()}</h3>")
    html_lines.append(f"<div class='meta'>pos_prob={r['pos_prob']} | final_score={r['final_score']} | flag_source={r['flag_source']} | reason={r.get('reason','')}</div>")
    # embed audio (relative path to Audios/ folder)
    html_lines.append(f"<audio controls src='{audio_rel}'></audio>")
    html_lines.append(f"<p><b>Transcript:</b> {transcript_high}</p>")
    if r["detected_keywords"]:
        html_lines.append(f"<p><b>Detected keywords:</b> {r['detected_keywords']}</p>")
    if r["fuzzy_diagnostics"]:
        html_lines.append(f"<p><b>Fuzzy diagnostics:</b> {r['fuzzy_diagnostics']}</p>")
    html_lines.append("</div>")

html_lines.append("<div class='footer'>Developed by Arnab Das — 2025</div>")
html_lines.append("</body></html>")

with open(html_path, "w", encoding="utf-8") as hf:
    hf.write("\\n".join(html_lines))

print("📄 HTML report saved to:", html_path)
print("📝 Log saved to:", log_path)
print("✅ Run complete.")
