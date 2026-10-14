"""
VEDA - All-in-one AI Voice Assistant
(GUI + Typing + Microphone + Gemini + Natural AI Voice + English/Hindi switch + Settings)

Install (ek baar):
    pip install customtkinter numpy google-genai edge-tts pyttsx3 psutil AppOpener plyer sounddevice faster-whisper

Run:
    python veda_app.py
"""
import asyncio
import json
import math
import os
import queue
import re
import sqlite3
import threading
import time
import tempfile
import ctypes
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

# ---- optional libraries (GUI tab bhi chalega agar koi missing ho) ----------
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    import psutil
except Exception:
    psutil = None
try:
    from AppOpener import open as launch_app
except Exception:
    launch_app = None
try:
    from plyer import notification
except Exception:
    notification = None
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------------------------------------------------------------------
# PATHS + CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "veda_config.json")
DB_PATH = os.path.join(BASE_DIR, "veda_privacy_vault.db")
TTS_FILE = os.path.join(tempfile.gettempdir(), "veda_tts.mp3")

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gemini-flash-latest",
    "reply_language": "Auto",
    "whisper_model": "base",
    "speak_replies": True,
    "use_search": True,
    "hands_free": False,
    "temperature": 0.6,
    "voice_rate": 170,
    "voice_gender": "Female",
}
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
LANG_CHOICES = ["Auto", "English", "Hindi"]

# Natural neural voices (Microsoft Edge online voices, free)
EDGE_VOICES = {
    ("Female", "hi"): "hi-IN-SwaraNeural",
    ("Female", "en"): "en-IN-NeerjaNeural",
    ("Male", "hi"): "hi-IN-MadhurNeural",
    ("Male", "en"): "en-IN-PrabhatNeural",
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    if not cfg.get("api_key"):
        cfg["api_key"] = os.environ.get("GEMINI_API_KEY", "")
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------
WAKE_RE = re.compile(
    r"(?:\b(?:hey|hay|hi|hello|ok|okay)\s+)?"
    r"(?:\b(?:veda|vedha|weda|vida|aveda|veeda)\b|वेदा|वीडा)",
    re.IGNORECASE,
)
EXIT_RE = re.compile(r"\b(exit|goodbye|quit|good bye)\b|अलविदा", re.IGNORECASE)

VOICE_HINTS = {
    "en": ["en-us", "en-in", "en-gb", "english"],
    "hi": ["hi-in", "hindi", "hemant", "kalpana"],
}


def has_devanagari(text):
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def clean_for_speech(text):
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"https?://\S+", " link ", t)
    t = re.sub(r"[*_#`>~|]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:900]


def is_model_error(e):
    """Model naam hi galat/band hai (permanent)."""
    s = str(e).lower()
    return "404" in s or "not found" in s or "no longer available" in s or "not supported" in s


def is_transient_error(e):
    """Google side busy/overloaded/timeout (thodi der me theek ho jata hai)."""
    s = str(e).lower()
    keys = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted",
            "500", "internal", "deadline", "timed out", "timeout", "502", "504")
    return any(k in s for k in keys)


def pick_best_model(client):
    """Key par available sabse stable naya flash model chunta hai."""
    best, best_score = None, None
    bad = ("tts", "image", "embedding", "live", "audio", "robotics", "computer", "learnlm",
           "vision", "exp", "gemma", "deep-research", "banana", "veo", "imagen", "aqa", "thinking")
    for m in client.models.list():
        acts = getattr(m, "supported_actions", None)
        if acts and "generateContent" not in acts:
            continue
        n = (m.name or "").replace("models/", "")
        low = n.lower()
        if not low.startswith("gemini") or any(x in low for x in bad):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", low)
        ver = float(v.group(1)) if v else 0.0
        score = ("flash" in low, "preview" not in low, ver, "lite" not in low, "latest" not in low)
        if best_score is None or score > best_score:
            best, best_score = n, score
    return best


def explain_error(e):
    s = str(e)
    low = s.lower()
    if "503" in s or "unavailable" in low or "high demand" in low or "overloaded" in low:
        return "Gemini abhi busy hai (Google ke server par load zyada hai). Thodi der baad phir try karo."
    if "api key" in low or "api_key" in low or "permission_denied" in low or "401" in s or "403" in s:
        return "API key galat ya invalid lagti hai. Settings tab me key check karo.\n(" + s[:150] + ")"
    if "429" in s or "quota" in low or "resource_exhausted" in low:
        return "Free quota/limit khatam ho gayi. Thodi der baad try karo, ya Settings me flash-lite model chuno."
    if "404" in s or "not found" in low:
        return "Ye model naam is key par available nahi hai. Settings me 'Fetch my models' dabake list se chuno."
    if "deadline" in low or "timed out" in low or "timeout" in low:
        return "Gemini ne time par jawab nahi diya (timeout). Internet check karo ya Settings me 'Test connection' dabao."
    if "getaddrinfo" in low or "connect" in low:
        return "Internet connection ki problem lag rahi hai."
    return "Error: " + s[:300]


def lerp_color(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    r = [int(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*r)


# ---------------------------------------------------------------------------
# TEXT TO SPEECH  (1: natural neural voice online, 2: pyttsx3 offline fallback)
# ---------------------------------------------------------------------------
_mixer_lock = threading.Lock()


async def _edge_save(text, voice, rate_str, path):
    comm = edge_tts.Communicate(text, voice, rate=rate_str)
    await comm.save(path)


def _mci(cmd):
    return ctypes.windll.winmm.mciSendStringW(cmd, None, 0, None)


def speak_neural(text, lang, rate, gender):
    """Natural AI voice (edge-tts + Windows ka built-in mp3 player). True = bol diya."""
    if edge_tts is None or os.name != "nt":
        return False
    try:
        voice = EDGE_VOICES.get((gender, lang)) or EDGE_VOICES[("Female", "en")]
        pct = int(round((int(rate) - 170) / 170 * 100))
        rate_str = f"{pct:+d}%"
        asyncio.run(_edge_save(text, voice, rate_str, TTS_FILE))
        with _mixer_lock:
            _mci("close vedatts")
            if _mci(f'open "{TTS_FILE}" type mpegvideo alias vedatts') != 0:
                return False
            _mci("play vedatts wait")
            _mci("close vedatts")
        return True
    except Exception as e:
        print("[Neural TTS error]", e)
        return False


def _pick_voice(engine, lang):
    hints = VOICE_HINTS.get(lang, [])
    for v in engine.getProperty("voices"):
        blob = f"{v.id} {v.name} {v.languages}".lower()
        if any(h in blob for h in hints):
            return v.id
    return None


def speak_offline(text, lang, rate):
    if pyttsx3 is None:
        return

    def worker():
        pythoncom = None
        try:
            try:
                import pythoncom  # Windows threads ke liye zaroori
                pythoncom.CoInitialize()
            except Exception:
                pythoncom = None
            engine = pyttsx3.init()
            engine.setProperty("rate", int(rate))
            engine.setProperty("volume", 1.0)
            vid = _pick_voice(engine, lang)
            if vid:
                engine.setProperty("voice", vid)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("[Offline TTS error]", e)
        finally:
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=60)


def speak_blocking(text, lang, rate, gender="Female"):
    text = clean_for_speech(text)
    if not text:
        return
    if not speak_neural(text, lang, rate, gender):
        speak_offline(text, lang, rate)


# ---------------------------------------------------------------------------
# PRIVACY VAULT (SQLite)
# ---------------------------------------------------------------------------
def init_vault():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS app_permissions (app_name TEXT PRIMARY KEY, granted_status INTEGER)")
    conn.commit()
    conn.close()


def db_get_permission(name):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT granted_status FROM app_permissions WHERE app_name = ?", (name.lower(),)).fetchone()
    conn.close()
    return None if row is None else bool(row[0])


def db_set_permission(name, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO app_permissions (app_name, granted_status) VALUES (?, ?)",
                 (name.lower(), 1 if status else 0))
    conn.commit()
    conn.close()


def ensure_permission(name, ask):
    st = db_get_permission(name)
    if st is not None:
        return st
    ok = ask(f"VEDA wants permission to control:\n\n   {name}\n\nAllow? (VEDA ye jawab yaad rakhega)")
    db_set_permission(name, ok)
    return ok


# ---------------------------------------------------------------------------
# LOCAL AUTOMATION (open apps, battery, time, reminders)
# ---------------------------------------------------------------------------
def parse_delay_seconds(cmd):
    m = re.search(r"(\d+)\s*(second|sec|minute|min|hour|hr|ghante|ghanta|मिनट|घंटे|सेकंड)", cmd)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if unit.startswith(("sec", "सेकंड")):
        return n
    if unit.startswith(("min", "मिनट")):
        return n * 60
    return n * 3600


def local_automation(command, ask, on_reminder):
    """Local command ho to reply text return karta hai, warna None (tab Gemini se poochho)."""
    c = command.lower().strip()

    if re.search(r"\b(routine|workflow|study mode)\b", c):
        if launch_app is None:
            return "AppOpener install nahi hai: pip install AppOpener"
        if not ensure_permission("multi step routine", ask):
            return "Permission denied for the routine."
        launch_app("whatsapp", match_closest=True, output=False)
        time.sleep(2)
        launch_app("visual studio code", match_closest=True, output=False)
        return "Study mode routine start ho gaya."

    if re.search(r"\b(remind|reminder|alarm|medicine)\b|याद दिला", c):
        if not ensure_permission("reminders and notifications", ask):
            return "Permission denied for reminders."
        delay = parse_delay_seconds(c)
        if delay is None:
            return "Time batao, jaise: remind me in 10 minutes to take medicine."
        threading.Timer(delay, on_reminder, args=(command,)).start()
        return f"Theek hai, {delay // 60} minute baad yaad dilaunga." if delay >= 60 else f"Theek hai, {delay} second baad yaad dilaunga."

    if "battery" in c or "बैटरी" in c:
        if psutil is None:
            return "psutil install nahi hai: pip install psutil"
        b = psutil.sensors_battery()
        if b is None:
            return "Battery sensor read nahi ho paya."
        return f"Battery {int(b.percent)} percent hai aur {'charging ho rahi hai' if b.power_plugged else 'battery par chal raha hai'}."

    if re.search(r"\b(what('s| is) the time|current time|time now)\b", c) or "कितने बजे" in c:
        return "Abhi time hai " + datetime.now().strftime("%I:%M %p")
    if re.search(r"\b(today'?s date|what('s| is) the date|what day)\b", c):
        return "Aaj " + datetime.now().strftime("%A, %d %B %Y") + " hai."

    m = re.search(r"\bopen\b\s+(.+)", c)
    if m:
        app = re.sub(r"\b(please|the|app|application)\b", "", m.group(1)).strip(" .,!?")
        if not app:
            return "Kaunsa app kholu?"
        if launch_app is None:
            return "AppOpener install nahi hai: pip install AppOpener"
        if not ensure_permission(app, ask):
            return f"{app} ke liye permission nahi mili."
        try:
            launch_app(app, match_closest=True, output=False, throw_error=True)
            return f"{app} khol raha hoon."
        except Exception as e:
            print("[Launch error]", e)
            return f"{app} nahi khul paya."

    return None


# ---------------------------------------------------------------------------
# GLOWING ORB WIDGET
# (NOTE: tkinter ke naamo jaise 'size' / 'state' ko overwrite nahi karte)
# ---------------------------------------------------------------------------
ACCENTS = {"idle": "#7C3AED", "listening": "#22C55E", "thinking": "#F59E0B", "speaking": "#38BDF8"}
PULSE = {"idle": 0.015, "listening": 0.07, "thinking": 0.05, "speaking": 0.09}


class Orb(tk.Canvas):
    def __init__(self, master, dim=170, bg="#181B2B"):
        super().__init__(master, width=dim, height=dim, bg=bg, highlightthickness=0)
        self.dim = dim
        self.n_rings = 7
        self.mode = "idle"
        self.phase = 0.0
        self.rings = [self.create_oval(0, 0, 1, 1, outline="", fill="#000000") for _ in range(self.n_rings)]
        self.star = self.create_polygon([0, 0] * 8, fill="#F5E9FF", outline="")
        self.redraw()

    def redraw(self):
        c = self.dim / 2
        scale = 1 + PULSE[self.mode] * math.sin(self.phase)
        r0 = c * 0.92
        accent = ACCENTS[self.mode]
        for i, ring in enumerate(self.rings):
            r = r0 * (1 - i * 0.13) * scale
            self.coords(ring, c - r, c - r, c + r, c + r)
            self.itemconfig(ring, fill=lerp_color("#1F1A3A", accent, (i / (self.n_rings - 1)) ** 0.8))
        big, small = 20 * scale, 6 * scale
        pts = []
        for k in range(8):
            ang = math.pi / 4 * k - math.pi / 2
            rad = big if k % 2 == 0 else small
            pts += [c + rad * math.cos(ang), c + rad * math.sin(ang)]
        self.coords(self.star, *pts)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
class VedaApp(ctk.CTk):
    STATUS = {
        "idle": "SAY SOMETHING",
        "listening": "LISTENING...",
        "thinking": "THINKING...",
        "speaking": "SPEAKING...",
    }

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        self.cfg = load_config()
        self.ui_q = queue.Queue()
        self.history = []
        self.client = None
        self.client_key = None
        self.stt = None
        self.stt_name = None
        self.busy = False
        self.search_off = False
        self.last_good = None
        self._cand_cache = None
        self.hands_free_on = False
        self.ui_state = "idle"          # FIX: pehle self.state tha jo tkinter ka function overwrite karta tha
        init_vault()

        self.title("VEDA - AI Voice Assistant")
        self.geometry("480x780")
        self.minsize(440, 680)
        self.configure(fg_color="#131524")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.tabs = ctk.CTkTabview(
            self, corner_radius=24, fg_color="#181B2B", segmented_button_fg_color="#0F111A",
            segmented_button_selected_color="#8B5CF6", segmented_button_selected_hover_color="#7C3AED",
            text_color="#CBD5E1",
        )
        self.tabs.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_live = self.tabs.add("VEDA Live")
        self.tab_vault = self.tabs.add("Security Vault")
        self.tab_settings = self.tabs.add("Settings")

        self.build_live()
        self.build_vault()
        self.build_settings()

        self.poll_ui()
        self.animate()

        if self.cfg["api_key"]:
            self.add_chat("VEDA", "Namaste! Main VEDA hoon. Kuch bhi type karo ya Microphone dabake bolo.")
        else:
            self.add_chat("VEDA", "Namaste! Pehle Settings tab me apni Gemini API key paste karke Save dabao, phir mujhse kuch bhi pucho.")
        if edge_tts is None:
            self.add_chat("VEDA", "Natural voice ke liye terminal me chalao:\npip install edge-tts\n(Tab tak robot voice chalegi.)")

    # ---------------- thread-safe UI helper ----------------
    def ui(self, fn, *args):
        self.ui_q.put((fn, args))

    def poll_ui(self):
        try:
            while True:
                fn, args = self.ui_q.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(50, self.poll_ui)

    def ask_yes_no(self, message):
        """Worker thread se safe popup; jawab milne tak wait karta hai."""
        result, ev = {}, threading.Event()

        def show():
            result["v"] = messagebox.askyesno("VEDA Permission", message, parent=self)
            ev.set()

        self.ui(show)
        ev.wait()
        return result["v"]

    # ---------------- LIVE TAB ----------------
    def build_live(self):
        t = self.tab_live
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(t, text="VEDA", text_color="#A78BFA",
                     font=ctk.CTkFont(size=34, weight="bold")).grid(row=0, column=0, pady=(8, 0))
        ctk.CTkLabel(t, text="YOUR PERSONAL AI ASSISTANT", text_color="#94A3B8",
                     font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0)
        self.lbl_status = ctk.CTkLabel(t, text=self.STATUS["idle"], text_color="#C4B5FD",
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_status.grid(row=2, column=0, pady=(10, 0))

        self.orb = Orb(t, dim=150)
        self.orb.grid(row=3, column=0, pady=8)

        self.chat = ctk.CTkTextbox(
            t, fg_color="#0F111A", text_color="#E2E8F0", corner_radius=16, border_width=1,
            border_color="#2E334D", wrap="word", font=ctk.CTkFont(size=14),
        )
        self.chat.grid(row=4, column=0, sticky="nsew", padx=12, pady=6)
        try:
            self.chat.tag_config("you", foreground="#C084FC")
            self.chat.tag_config("veda", foreground="#60A5FA")
        except Exception:
            self.chat._textbox.tag_config("you", foreground="#C084FC")
            self.chat._textbox.tag_config("veda", foreground="#60A5FA")
        self.chat.configure(state="disabled")

        row = ctk.CTkFrame(t, fg_color="transparent")
        row.grid(row=5, column=0, sticky="ew", padx=12, pady=6)
        row.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(row, placeholder_text="Type your command here...", height=42,
                                  fg_color="#0F111A", border_color="#2E334D", corner_radius=14)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self.on_send())
        ctk.CTkButton(row, text="Send", width=80, height=42, fg_color="#2563EB", hover_color="#1D4ED8",
                      corner_radius=14, command=self.on_send).grid(row=0, column=1)

        btns = ctk.CTkFrame(t, fg_color="transparent")
        btns.grid(row=6, column=0, pady=(2, 4))
        self.btn_mic = ctk.CTkButton(btns, text="Microphone", width=150, height=42, fg_color="#8B5CF6",
                                     hover_color="#7C3AED", corner_radius=14, command=self.on_mic)
        self.btn_mic.grid(row=0, column=0, padx=5)
        ctk.CTkButton(btns, text="Clear", width=80, height=42, fg_color="#0F111A", hover_color="#24273A",
                      corner_radius=14, command=self.clear_chat).grid(row=0, column=1, padx=5)
        ctk.CTkButton(btns, text="Settings", width=100, height=42, fg_color="#0F111A", hover_color="#24273A",
                      corner_radius=14, command=lambda: self.tabs.set("Settings")).grid(row=0, column=2, padx=5)

        # quick controls: reply language + voice on/off
        quick = ctk.CTkFrame(t, fg_color="transparent")
        quick.grid(row=7, column=0, pady=(0, 8))
        ctk.CTkLabel(quick, text="Reply in:", text_color="#94A3B8",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, padx=(0, 6))
        self.quick_lang = ctk.CTkOptionMenu(quick, values=LANG_CHOICES, width=110, fg_color="#0F111A",
                                            button_color="#8B5CF6", command=self.on_quick_lang)
        self.quick_lang.grid(row=0, column=1, padx=(0, 14))
        self.quick_lang.set(self.cfg["reply_language"])
        self.sw_quick_voice = ctk.CTkSwitch(quick, text="Voice", command=self.on_quick_voice)
        self.sw_quick_voice.grid(row=0, column=2)
        if self.cfg["speak_replies"]:
            self.sw_quick_voice.select()
        self.refresh_mic_button()

    def on_quick_lang(self, value):
        self.cfg["reply_language"] = value
        save_config(self.cfg)
        if hasattr(self, "opt_lang"):
            self.opt_lang.set(value)

    def on_quick_voice(self):
        on = bool(self.sw_quick_voice.get())
        self.cfg["speak_replies"] = on
        save_config(self.cfg)
        if hasattr(self, "sw_speak"):
            self.sw_speak.select() if on else self.sw_speak.deselect()

    def add_chat(self, who, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", who + "\n", "you" if who == "You" else "veda")
        self.chat.insert("end", text + "\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.history.clear()

    def set_state(self, state):
        self.ui_state = state
        self.orb.mode = state
        text = self.STATUS[state]
        if state == "idle" and self.hands_free_on:
            text = "SAY 'HEY VEDA'"
        self.lbl_status.configure(text=text)

    def set_status_text(self, text):
        self.lbl_status.configure(text=text)

    def animate(self):
        self.orb.phase += 0.08 if self.ui_state == "idle" else 0.3
        self.orb.redraw()
        self.after(60, self.animate)

    def refresh_mic_button(self):
        if self.cfg["hands_free"]:
            self.btn_mic.configure(text="Stop hands-free" if self.hands_free_on else "Start hands-free")
        else:
            self.btn_mic.configure(text="Microphone")

    def finish(self):
        self.busy = False
        self.set_state("idle")

    # ---------------- typing ----------------
    def on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        if self.busy:
            self.set_status_text("WAIT... BUSY")
            return
        self.entry.delete(0, "end")
        self.busy = True
        self.set_state("thinking")
        threading.Thread(target=self._typed_worker, args=(text,), daemon=True).start()

    def _typed_worker(self, text):
        try:
            self._run_command(text)
        finally:
            self.ui(self.finish)

    # ---------------- shared pipeline ----------------
    def voice_lang_for(self, reply):
        setting = self.cfg["reply_language"]
        if setting == "English":
            return "en"
        if setting == "Hindi":
            return "hi"
        return "hi" if has_devanagari(reply) else "en"

    def _run_command(self, text):
        self.ui(self.add_chat, "You", text)
        self.ui(self.set_state, "thinking")
        failed = False
        try:
            reply = local_automation(text, self.ask_yes_no, self.on_reminder)
            if reply is None:
                reply = self.ask_gemini(text)
            self.history.append(("user", text))
            self.history.append(("model", reply))
            self.history = self.history[-16:]
        except Exception as e:
            print("[VEDA error]", repr(e))
            reply = "⚠️ " + explain_error(e)
            failed = True
        self.ui(self.add_chat, "VEDA", reply)
        if self.cfg["speak_replies"]:
            self.ui(self.set_state, "speaking")
            try:
                if failed:
                    # error par bhi awaaz me batao (chhota sa)
                    speak_blocking("Sorry, I could not get an answer right now. Please try again in a moment.",
                                   "en", self.cfg["voice_rate"], self.cfg.get("voice_gender", "Female"))
                else:
                    speak_blocking(reply, self.voice_lang_for(reply),
                                   self.cfg["voice_rate"], self.cfg.get("voice_gender", "Female"))
            except Exception as e:
                print("[Speak error]", e)

    def on_reminder(self, message):
        self.ui(self.add_chat, "VEDA", f"⏰ Reminder: {message}")
        if notification:
            try:
                notification.notify(title="VEDA Reminder", message=message, app_name="VEDA", timeout=10)
            except Exception:
                pass

    # ---------------- Gemini ----------------
    def get_client(self):
        if genai is None:
            raise RuntimeError("google-genai install nahi hai: pip install google-genai")
        key = self.cfg["api_key"].strip()
        if not key:
            raise RuntimeError("API key missing hai. Settings tab me key paste karke Save dabao.")
        if self.client is None or self.client_key != key:
            try:
                # 15 sec timeout: "THINKING..." par atakega nahi
                self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=15000))
            except Exception:
                self.client = genai.Client(api_key=key)
            self.client_key = key
        return self.client

    def system_prompt(self):
        lang = self.cfg["reply_language"]
        if lang == "Hindi":
            rule = "Reply only in Hindi (Devanagari script)."
        elif lang == "English":
            rule = "Reply only in English, even if the user writes in Hindi or Hinglish."
        else:
            rule = ("Reply in the same language the user used. If the user writes Hindi in Roman letters "
                    "(Hinglish), reply in Hinglish; if in Devanagari, reply in Devanagari Hindi.")
        return (
            "You are VEDA, a friendly and smart personal AI assistant. "
            "Solve math step by step, explain things simply, help with studies, coding and general knowledge. "
            "For news, weather or current events use search results; if you cannot get real data, say so honestly "
            "and never invent facts. " + rule + " Keep answers clear and reasonably short because they will be "
            "spoken aloud. Avoid heavy markdown, bullet symbols and emojis."
        )

    def get_candidates(self, client):
        """Is key par available Gemini models (best pehle). Cache hota hai."""
        if self._cand_cache is None:
            bad = ("tts", "image", "embedding", "live", "audio", "robotics", "computer", "learnlm",
                   "vision", "exp", "gemma", "deep-research", "banana", "veo", "imagen", "aqa", "thinking")
            found = []
            try:
                for m in client.models.list():
                    acts = getattr(m, "supported_actions", None)
                    if acts and "generateContent" not in acts:
                        continue
                    n = (m.name or "").replace("models/", "")
                    low = n.lower()
                    if not low.startswith("gemini") or any(x in low for x in bad):
                        continue
                    v = re.search(r"gemini-(\d+(?:\.\d+)?)", low)
                    ver = float(v.group(1)) if v else 0.0
                    found.append((("flash" in low, "preview" not in low, ver, "lite" not in low, "latest" not in low), n))
            except Exception as e:
                print("[Model list failed]", repr(e)[:150])
            found.sort(reverse=True)
            self._cand_cache = [n for _, n in found]
        return self._cand_cache

    def call_model(self, client, model, contents):
        def gen(use_search):
            kwargs = dict(system_instruction=self.system_prompt(), temperature=float(self.cfg["temperature"]))
            if use_search:
                kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            return client.models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**kwargs))

        if self.cfg["use_search"] and not self.search_off:
            try:
                return gen(True)
            except Exception as e:
                low = str(e).lower()
                if is_model_error(e):
                    raise
                if "429" in low or "resource_exhausted" in low or "quota" in low:
                    # Google Search ka free quota khatam: ab session me search band
                    self.search_off = True
                    print("[Search quota over -> search OFF]", repr(e)[:120])
                    self.ui(self.add_chat, "VEDA", "Google Search ka quota khatam hai, ab bina search ke jawab dunga.")
                else:
                    raise
        return gen(False)

    def ask_gemini(self, text):
        client = self.get_client()
        contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in self.history[-12:]]
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

        listed = self.get_candidates(client)
        primary = self.cfg["model"].strip() or "gemini-flash-latest"
        order = []
        if self.last_good:
            order.append(self.last_good)
        if primary in listed or not listed:
            order.append(primary)
        order += listed[:5]
        if not listed:
            order += FALLBACK_MODELS
        seen, models = set(), []
        for m in order:
            if m not in seen:
                seen.add(m)
                models.append(m)
        models = models[:4]

        last_err, resp, used = None, None, None
        for model in models:
            try:
                resp = self.call_model(client, model, contents)
                used = model
                break
            except Exception as e:
                last_err = e
                print(f"[{model} failed]", repr(e)[:200])
                if is_model_error(e) or is_transient_error(e):
                    continue          # agla model try karo
                raise                 # key galat etc.
        if resp is None:
            raise last_err
        self.last_good = used
        if used != primary and listed and primary not in listed:
            self.cfg["model"] = used          # purana model band hai: naya save karo
            save_config(self.cfg)
            self.ui(lambda m=used: self.cmb_model.set(m))
        return (resp.text or "").strip() or "Mujhe is par jawab nahi mila."

    # ---------------- microphone ----------------
    def mic_ready(self):
        if sd is None or WhisperModel is None:
            self.add_chat("VEDA", "Mic libraries missing hain. Terminal me chalao:\npip install sounddevice faster-whisper")
            return False
        return True

    def on_mic(self):
        if not self.mic_ready():
            return
        if self.cfg["hands_free"]:
            if self.hands_free_on:
                self.hands_free_on = False
                self.refresh_mic_button()
                self.set_state("idle")
            else:
                self.hands_free_on = True
                self.refresh_mic_button()
                self.set_state("listening")
                threading.Thread(target=self._hands_free_loop, daemon=True).start()
        else:
            if self.busy:
                return
            self.busy = True
            self.set_state("listening")
            threading.Thread(target=self._mic_once, daemon=True).start()

    def record_speech(self, max_wait=8.0):
        """Bolna shuru hone par record start, 1.4 sec khamoshi par stop. Kuch na bola to None."""
        sr, chunk = 16000, 1280
        sec = chunk / sr
        max_silent = int(1.4 / sec)
        max_wait_n = int(max_wait / sec)
        max_total = int(25 / sec)
        preroll = deque(maxlen=5)
        rec, started, loud, silent, waited = [], False, 0, 0, 0
        floor = 1e9
        with sd.InputStream(samplerate=sr, channels=1, blocksize=chunk, dtype="int16") as stream:
            while True:
                frame, _ = stream.read(chunk)
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                floor = min(floor, max(rms, 50.0))
                thr = max(300.0, floor * 2.5)
                if not started:
                    preroll.append(frame)
                    loud = loud + 1 if rms > thr else 0
                    if loud >= 2:
                        started = True
                        rec = list(preroll)
                    waited += 1
                    if waited >= max_wait_n:
                        return None
                else:
                    rec.append(frame)
                    silent = silent + 1 if rms < thr else 0
                    if silent >= max_silent or len(rec) >= max_total:
                        break
        return np.concatenate(rec).flatten().astype(np.float32) / 32768.0

    def get_stt(self):
        name = self.cfg["whisper_model"]
        if self.stt is None or self.stt_name != name:
            self.ui(self.set_status_text, f"LOADING SPEECH MODEL '{name}'...")
            self.stt = WhisperModel(name, device="cpu", compute_type="int8")
            self.stt_name = name
        return self.stt

    def transcribe(self, audio, prompt=None):
        stt = self.get_stt()
        opts = dict(beam_size=1, vad_filter=True, condition_on_previous_text=False)
        if prompt:
            opts["initial_prompt"] = prompt
        segments, info = stt.transcribe(audio, **opts)
        text = " ".join(s.text for s in segments).strip()
        lang = info.language
        if lang not in ("en", "hi") or info.language_probability < 0.5:
            segments, _ = stt.transcribe(audio, language="en", **opts)
            text = " ".join(s.text for s in segments).strip()
            lang = "en"
        return text, lang

    def _mic_once(self):
        try:
            audio = self.record_speech(max_wait=8.0)
            if audio is None:
                self.ui(self.add_chat, "VEDA", "Kuch sunai nahi diya. Microphone dabake phir se bolo.")
                return
            self.ui(self.set_state, "thinking")
            text, _lang = self.transcribe(audio)
            if len(text) <= 1:
                self.ui(self.add_chat, "VEDA", "Samajh nahi aaya, ek baar phir bolo.")
                return
            self._run_command(text)
        except Exception as e:
            self.ui(self.add_chat, "VEDA", "⚠️ Mic error: " + str(e)[:250])
        finally:
            self.ui(self.finish)

    def _hands_free_loop(self):
        try:
            while self.hands_free_on:
                if self.busy:
                    time.sleep(0.3)
                    continue
                self.ui(self.set_state, "listening")
                audio = self.record_speech(max_wait=10.0)
                if audio is None or not self.hands_free_on:
                    continue
                text, _lang = self.transcribe(audio, prompt="Hey Veda")
                m = WAKE_RE.search(text) if text else None
                if not m:
                    continue
                command = (text[:m.start()] + " " + text[m.end():]).strip(" ,.?!")
                if EXIT_RE.search(command):
                    self.hands_free_on = False
                    self.ui(self.add_chat, "VEDA", "Hands-free mode band. Goodbye!")
                    break
                if len(command) <= 1:
                    command = "hello"
                self.busy = True
                try:
                    self._run_command(command)
                finally:
                    self.busy = False
        except Exception as e:
            self.ui(self.add_chat, "VEDA", "⚠️ Hands-free error: " + str(e)[:250])
        finally:
            self.hands_free_on = False
            self.ui(self.refresh_mic_button)
            self.ui(self.set_state, "idle")

    # ---------------- VAULT TAB ----------------
    def build_vault(self):
        t = self.tab_vault
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(t, text="App Permission Records", text_color="#A78BFA",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))
        self.vault_box = ctk.CTkTextbox(t, fg_color="#0F111A", text_color="#C084FC", border_width=1,
                                        border_color="#2E334D", corner_radius=12,
                                        font=ctk.CTkFont(family="Courier New", size=13))
        self.vault_box.grid(row=1, column=0, sticky="nsew", padx=15, pady=8)
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.grid(row=2, column=0, pady=10)
        ctk.CTkButton(row, text="Refresh", fg_color="#2563EB", hover_color="#1D4ED8",
                      command=self.refresh_vault).grid(row=0, column=0, padx=6)
        ctk.CTkButton(row, text="Reset all permissions", fg_color="#7F1D1D", hover_color="#991B1B",
                      command=self.reset_vault).grid(row=0, column=1, padx=6)
        self.refresh_vault()

    def refresh_vault(self):
        self.vault_box.configure(state="normal")
        self.vault_box.delete("1.0", "end")
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("SELECT app_name, granted_status FROM app_permissions ORDER BY app_name").fetchall()
            conn.close()
            if not rows:
                self.vault_box.insert("end", "Abhi koi permission record nahi hai.\n\nJab VEDA pehli baar koi app kholega\nya reminder lagayega to yahan dikhega.")
            for name, status in rows:
                self.vault_box.insert("end", f"• {name.upper()}  ->  {'GRANTED' if status == 1 else 'DENIED'}\n")
        except Exception as e:
            self.vault_box.insert("end", "Read error: " + str(e))
        self.vault_box.configure(state="disabled")

    def reset_vault(self):
        if messagebox.askyesno("Reset", "Saari saved permissions delete kar du?", parent=self):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM app_permissions")
            conn.commit()
            conn.close()
            self.refresh_vault()

    # ---------------- SETTINGS TAB ----------------
    def build_settings(self):
        t = self.tab_settings
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(0, weight=1)
        f = ctk.CTkScrollableFrame(t, fg_color="transparent")
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)

        def head(text):
            ctk.CTkLabel(f, text=text, text_color="#A78BFA", font=ctk.CTkFont(size=14, weight="bold"),
                         anchor="w").pack(fill="x", padx=12, pady=(14, 2))

        def hint(text):
            ctk.CTkLabel(f, text=text, text_color="#64748B", font=ctk.CTkFont(size=11), anchor="w",
                         justify="left", wraplength=380).pack(fill="x", padx=12)

        head("Gemini API Key")
        hint("aistudio.google.com/apikey se free key banao. Key sirf is PC par veda_config.json me save hoti hai.")
        self.ent_key = ctk.CTkEntry(f, show="•", height=38, fg_color="#0F111A", border_color="#2E334D")
        self.ent_key.pack(fill="x", padx=12, pady=4)
        self.ent_key.insert(0, self.cfg["api_key"])
        self.sw_show = ctk.CTkSwitch(f, text="Show key", command=lambda: self.ent_key.configure(
            show="" if self.sw_show.get() else "•"))
        self.sw_show.pack(anchor="w", padx=12, pady=2)

        head("Gemini Model / Version")
        hint("flash = fast + free tier ke liye best. Busy hone par VEDA khud dusra model try karta hai. "
             "Naya model chahiye to 'Fetch my models' dabao ya naam khud type karo.")
        self.cmb_model = ctk.CTkComboBox(f, values=FALLBACK_MODELS, height=38, fg_color="#0F111A",
                                         border_color="#2E334D")
        self.cmb_model.pack(fill="x", padx=12, pady=4)
        self.cmb_model.set(self.cfg["model"])
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", padx=12, pady=2)
        ctk.CTkButton(r, text="Fetch my models", width=140, command=self.fetch_models).pack(side="left", padx=(0, 6))
        ctk.CTkButton(r, text="Test connection", width=140, fg_color="#059669", hover_color="#047857",
                      command=self.test_connection).pack(side="left")

        head("Reply Language")
        hint("Auto = jis bhasha me poochho usi me jawab. English / Hindi = hamesha wahi bhasha.")
        self.opt_lang = ctk.CTkOptionMenu(f, values=LANG_CHOICES, fg_color="#0F111A",
                                          button_color="#8B5CF6", command=self.on_quick_lang_from_settings)
        self.opt_lang.pack(anchor="w", padx=12, pady=4)
        self.opt_lang.set(self.cfg["reply_language"])

        head("AI Voice")
        hint("Natural AI voice ke liye internet chahiye (edge-tts). Na chale to offline robot voice use hoti hai.")
        self.opt_gender = ctk.CTkOptionMenu(f, values=["Female", "Male"], fg_color="#0F111A",
                                            button_color="#8B5CF6")
        self.opt_gender.pack(anchor="w", padx=12, pady=4)
        self.opt_gender.set(self.cfg.get("voice_gender", "Female"))
        ctk.CTkButton(f, text="Test voice", width=140, fg_color="#0EA5E9", hover_color="#0284C7",
                      command=self.test_voice).pack(anchor="w", padx=12, pady=4)

        head("Speech Recognition Model (microphone)")
        hint("tiny = fastest, base = balanced, small = Hindi ke liye best (slow, pehli baar download hoga).")
        self.opt_whisper = ctk.CTkOptionMenu(f, values=["tiny", "base", "small"], fg_color="#0F111A",
                                             button_color="#8B5CF6")
        self.opt_whisper.pack(anchor="w", padx=12, pady=4)
        self.opt_whisper.set(self.cfg["whisper_model"])

        head("Options")
        self.sw_speak = ctk.CTkSwitch(f, text="Speak replies aloud", command=self.on_settings_voice_switch)
        self.sw_speak.pack(anchor="w", padx=12, pady=4)
        self.sw_search = ctk.CTkSwitch(f, text="Use Google Search for live info (news, weather)")
        self.sw_search.pack(anchor="w", padx=12, pady=4)
        self.sw_hands = ctk.CTkSwitch(f, text="Hands-free mode (say 'Hey Veda ...')")
        self.sw_hands.pack(anchor="w", padx=12, pady=4)
        for sw, key in ((self.sw_speak, "speak_replies"), (self.sw_search, "use_search"), (self.sw_hands, "hands_free")):
            if self.cfg[key]:
                sw.select()

        head("Creativity (temperature)")
        self.sld_temp = ctk.CTkSlider(f, from_=0, to=1, number_of_steps=20)
        self.sld_temp.pack(fill="x", padx=12, pady=4)
        self.sld_temp.set(self.cfg["temperature"])

        head("Voice Speed")
        self.sld_rate = ctk.CTkSlider(f, from_=120, to=230, number_of_steps=22)
        self.sld_rate.pack(fill="x", padx=12, pady=4)
        self.sld_rate.set(self.cfg["voice_rate"])

        ctk.CTkButton(f, text="Save Settings", height=42, fg_color="#8B5CF6", hover_color="#7C3AED",
                      command=self.on_save_settings).pack(fill="x", padx=12, pady=(18, 4))
        self.lbl_settings = ctk.CTkLabel(f, text="", text_color="#94A3B8", wraplength=380, justify="left")
        self.lbl_settings.pack(fill="x", padx=12, pady=(2, 20))

    def on_quick_lang_from_settings(self, value):
        self.quick_lang.set(value)

    def on_settings_voice_switch(self):
        if self.sw_speak.get():
            self.sw_quick_voice.select()
        else:
            self.sw_quick_voice.deselect()

    def apply_settings(self):
        c = self.cfg
        c["api_key"] = self.ent_key.get().strip()
        c["model"] = self.cmb_model.get().strip() or "gemini-flash-latest"
        c["reply_language"] = self.opt_lang.get()
        c["voice_gender"] = self.opt_gender.get()
        c["whisper_model"] = self.opt_whisper.get()
        c["speak_replies"] = bool(self.sw_speak.get())
        c["use_search"] = bool(self.sw_search.get())
        c["hands_free"] = bool(self.sw_hands.get())
        c["temperature"] = round(float(self.sld_temp.get()), 2)
        c["voice_rate"] = int(self.sld_rate.get())
        save_config(c)
        if not c["hands_free"]:
            self.hands_free_on = False
        self.quick_lang.set(c["reply_language"])
        if c["speak_replies"]:
            self.sw_quick_voice.select()
        else:
            self.sw_quick_voice.deselect()
        self.refresh_mic_button()

    def on_save_settings(self):
        self.apply_settings()
        self.lbl_settings.configure(text="✅ Settings save ho gayi.", text_color="#34D399")

    def test_voice(self):
        self.apply_settings()
        if edge_tts is None:
            self.lbl_settings.configure(
                text="⚠️ Natural voice ke liye terminal me chalao:\npip install edge-tts\nTab tak robot voice chalegi.",
                text_color="#F87171")
        else:
            self.lbl_settings.configure(text="Voice test chal raha hai...", text_color="#94A3B8")

        def work():
            try:
                lang = "hi" if self.cfg["reply_language"] == "Hindi" else "en"
                msg = ("नमस्ते! मैं वेदा हूँ, आपकी पर्सनल AI असिस्टेंट।" if lang == "hi"
                       else "Hello! I am Veda, your personal AI assistant.")
                speak_blocking(msg, lang, self.cfg["voice_rate"], self.cfg.get("voice_gender", "Female"))
            except Exception as e:
                print("[Test voice error]", e)

        threading.Thread(target=work, daemon=True).start()

    def fetch_models(self):
        self.apply_settings()
        self.lbl_settings.configure(text="Models fetch ho rahe hain...", text_color="#94A3B8")

        def work():
            try:
                client = self.get_client()
                names = []
                for m in client.models.list():
                    acts = getattr(m, "supported_actions", None)
                    if acts and "generateContent" not in acts:
                        continue
                    n = (m.name or "").replace("models/", "")
                    if n.startswith("gemini"):
                        names.append(n)
                names = sorted(set(names))
                if not names:
                    raise RuntimeError("Koi model nahi mila.")
                self.ui(lambda: self.cmb_model.configure(values=names))
                self.ui(lambda: self.lbl_settings.configure(
                    text=f"✅ {len(names)} models mile. Dropdown se chuno, phir Save dabao.", text_color="#34D399"))
            except Exception as e:
                msg = "⚠️ " + explain_error(e)
                self.ui(lambda: self.lbl_settings.configure(text=msg, text_color="#F87171"))

        threading.Thread(target=work, daemon=True).start()

    def test_connection(self):
        self.apply_settings()
        self.lbl_settings.configure(text="Test chal raha hai...", text_color="#94A3B8")

        def work():
            try:
                client = self.get_client()
                self._cand_cache = None
                listed = self.get_candidates(client)
                cur = self.cfg["model"]
                order = [cur] + [m for m in listed[:4] if m != cur]
                last = None
                for m in order:
                    try:
                        resp = client.models.generate_content(model=m, contents="Reply with the single word: OK")
                        if m != cur and listed and cur not in listed:
                            self.cfg["model"] = m
                            save_config(self.cfg)
                            self.ui(lambda mm=m: self.cmb_model.set(mm))
                        out = (resp.text or "").strip()[:40]
                        self.ui(lambda mm=m, o=out: self.lbl_settings.configure(
                            text=f"✅ Connected! Model '{mm}' ne jawab diya: {o}", text_color="#34D399"))
                        return
                    except Exception as e:
                        last = e
                        if is_model_error(e) or is_transient_error(e):
                            continue
                        raise
                raise last
            except Exception as e:
                msg = "⚠️ " + explain_error(e)
                self.ui(lambda: self.lbl_settings.configure(text=msg, text_color="#F87171"))

        threading.Thread(target=work, daemon=True).start()

    def on_close(self):
        self.hands_free_on = False
        self.destroy()


if __name__ == "__main__":
    VedaApp().mainloop()