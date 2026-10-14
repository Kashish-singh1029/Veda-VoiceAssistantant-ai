# VEDA - All-in-One AI Voice Assistant

**VEDA** is an advanced AI assistant featuring a Graphical User Interface (GUI) built with CustomTkinter. It integrates voice, text typing, microphone input, and Google's Gemini AI to offer a fluid conversational experience supporting both English and Hindi language switching.

---

## 🛠 Features

* **Multimodal UI:** Clean, responsive desktop interface using `customtkinter`.
* **Smart Interaction:** Supports continuous voice interaction, text typing, and real-time audio playback.
* **Powered by Gemini AI:** Uses the `google-genai` engine for smart reasoning and contextual responses.
* **Language Switch:** Seamlessly toggle between English and Hindi assistance.
* **System Utilities:** Background resource handling using `psutil`, notifications via `plyer`, and audio management via `pygame` / `sounddevice`.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Installation (Run Once)
Open your terminal or command prompt inside the project directory and execute the following command to install all the necessary backend dependencies:

```bash
pip install customtkinter numpy google-genai edge-tts pyttsx3 psutil AppOpener plyer sounddevice pygame
```

### 2. Running the Application
To launch the VEDA Voice Assistant interface, run your main application script via the terminal:

```bash
python veda_app.py
```

---

## 📦 Core Project Architecture

* `veda_app.py` - Core GUI and application lifecycle entry point.
* `veda_config.json` - System configuration settings.
* `veda_privacy_vault.db` - Local storage SQLite data wrapper for managing secure permissions.

---
## ⚠️ Troubleshooting

### Gemini 503 Error (Service Unavailable)
If you see a `503 Service Unavailable` or `Spikes in demand are usually temporary` error in your terminal console:

* **Why it happens:** The Gemini API endpoint is currently experiencing high traffic volumes or rate-limiting restrictions.
* **How to fix:** 
  1. Wait a few moments and try your command again.
  2. Implement an exponential backoff retry loop within your Python code.

## 📄 License
This project is open-source and available under the standard MIT License.
