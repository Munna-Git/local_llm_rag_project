

# 🧠 RAG Project Setup — Difficulties & Solutions

## ⚙️ 1️⃣ Docker command not running properly (Windows new-line issue)

**Problem:**
When running:

```bash
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.19.2
```

Windows Command Prompt treated each line as a new command.

**Cause:**
Windows CMD doesn’t recognize the `\` line continuation used in Linux/macOS shells.

**Solution:**
Run it as **one single line**:

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.19.2
```

✅ Worked correctly afterward.

---

## 📦 2️⃣ `ollama run qwen3:8b` download failed (connection timeout)

**Problem:**
Download failed midway with an error about connection timeout to Cloudflare storage.

**Cause:**
This happens due to **network instability**, **firewalls**, or **Cloudflare’s region-based throttling** during large pulls.

**Solution:**

* Retried with a **stable internet connection**.
* If it persists, use a **VPN** or run Ollama with `--timeout` or resume support in a **Linux WSL environment**.
* The issue isn’t your setup — it’s external network reachability.

---

## 🧱 3️⃣ Tesseract not recognized after installation

**Problem:**
After running:

```bash
winget install UB-Mannheim.TesseractOCR
```

the command:

```bash
tesseract --version
```

gave:

```
'tesseract' is not recognized as an internal or external command
```

**Cause:**
Tesseract was installed, but its path (`C:\Program Files\Tesseract-OCR`) wasn’t added to the system PATH.

**Solution:**

* Added this manually to Environment Variables → PATH,
  **or** used this in Python:

  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

✅ Verified with `tesseract --version` → worked.

## Add it to PATH:

Press Windows key → type "Environment Variables" → Edit system environment variables.

Click Environment Variables…

Under System variables, select Path → Edit → New

Paste the folder path:

C:\Program Files\Tesseract-OCR


Click OK on all windows to save.

Important: Close your terminal and open a new Command Prompt or PowerShell. PATH changes don’t affect already-open terminals.

---

## 📄 4️⃣ Poppler “Unable to get page count” / I/O Error

**Problem:**

```python
pages = convert_from_path("resume.pdf", poppler_path=...)
```

threw:

```
PDFPageCountError: Unable to get page count. I/O Error: Couldn't open file
```

**Cause:**

* PDF path was incorrect or didn’t exist.
* Sometimes quotes or wrong slashes in Windows paths cause this.

**Solution:**
Used full absolute path and raw string literal:

```python
pdf_path = r"C:\Users\MUNNA\Downloads\resume_Sanchia.pdf"
pages = convert_from_path(pdf_path, poppler_path=r"C:\poppler\poppler-25.07.0\Library\bin")
```

✅ Worked once the correct path was given.

## Add Poppler to PATH

Press Windows key → type "Environment Variables" → Edit system environment variables.

Click Environment Variables…

Under System variables, select Path → Edit → New

Paste the bin folder path, e.g.:

C:\poppler\poppler-24.08.0\bin


Click OK → OK → OK to save.

Important: Open a new terminal after this — PATH changes don’t apply to already-open windows.

---

## 🔍 5️⃣ TesseractNotFoundError in Python

**Problem:**
`pytesseract.image_to_string()` threw:

```
TesseractNotFoundError: tesseract is not installed or it's not in your PATH
```

**Cause:**
`pytesseract` couldn’t locate the Tesseract executable automatically.

**Solution:**
Explicitly set the path:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

✅ OCR worked successfully afterward.

---

## 🧠 6️⃣ Confusion: Why do we need PATH / .exe / “binaries”?

**Problem:**
You asked why we have to “set the path” and what `.exe` or “binaries” even mean.

**Cause:**
It’s not obvious until you understand the system-level interaction between Python (a high-level interpreter) and native C++ executables like Tesseract or Poppler.

**Solution (Understanding):**

* `.exe` → executable file in Windows.
* “Binaries” → precompiled machine code programs.
* PATH → system list of folders where OS looks for programs.
* Python libraries like `pytesseract` are *wrappers* around these executables, not replacements.
  ✅ Once PATH was set or specified, Python could find and run Tesseract.

---

# ✅ Final Setup Summary

| Component           | Installed | Verified                                        |
| ------------------- | --------- | ----------------------------------------------- |
| **Docker**          | ✅         | Runs OpenSearch + Dashboards                    |
| **OpenSearch**      | ✅         | Dashboard accessible at `http://localhost:5601` |
| **Tesseract OCR**   | ✅         | `tesseract --version` works                     |
| **Poppler**         | ✅         | `convert_from_path()` works                     |
| **Python OCR Test** | ✅         | Extracts text from PDF                          |
| **System PATH**     | ✅         | Tesseract found globally                        |

---

# 💡 Key Takeaways

1. **Windows vs Linux line continuations** matter in Docker commands.
2. **PATH environment variable** determines where programs are found.
3. Python OCR tools rely on **native executables (binaries)**.
4. Always use **absolute paths** for files in Windows.
5. Understanding what’s happening under the hood prevents “black box” frustration.
"# local_llm_rag_project" 
