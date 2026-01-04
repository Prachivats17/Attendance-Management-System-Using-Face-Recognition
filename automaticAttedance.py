# ---------------- AUTOMATIC ATTENDANCE MODULE (MAC SAFE) ------------------

import os
import tkinter as tk
from tkinter import ttk, messagebox
import time
import datetime
import pandas as pd
import cv2

# -------------------- PATHS ---------------------
HAAR_PATH = "haarcascade_frontalface_default.xml"
TRAIN_LABEL_PATH = "./TrainingImageLabel/Trainner.yml"
studentdetail_path = "./StudentDetails/studentdetails.csv"
attendance_path = "./Attendance"

os.makedirs(attendance_path, exist_ok=True)


# ===================================================================
#                  SUBJECT CHOOSE + ATTENDANCE
# ===================================================================

def subjectChoose(text_to_speech):

    # ---------------- COLORS ----------------
    BG = "#0f1113"
    CARD = "#181a1d"
    ACCENT = "#f2d300"
    TEXT = "#e6e6e6"
    MUTED = "#979ca3"
    BTN_BG = "#111214"

    # ---------- WINDOW ----------
    win = tk.Toplevel()
    win.title("Take Attendance — CLASS VISION")
    win.configure(bg=BG)
    win.resizable(False, False)
    win.geometry("600x380")

    # ------------- CENTER WINDOW -------------
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"+{(sw//2)-300}+{(sh//2)-190}")

    tk.Label(win, text="Enter Subject Name", bg=BG, fg=ACCENT,
             font=("Segoe UI", 20, "bold")).pack(pady=(12, 8))

    tk.Label(win, text="Subject name will be used to save the attendance file.",
             bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(pady=(0, 12))

    # ENTRY CARD
    card = tk.Frame(win, bg=CARD)
    card.pack(fill="x", padx=20, pady=6)

    tk.Label(card, text="Subject", bg=CARD, fg=TEXT,
             font=("Segoe UI", 12)).pack(anchor="w", padx=14, pady=(12, 4))

    entry = tk.Entry(card, bg=BTN_BG, fg=ACCENT,
                     insertbackground=ACCENT,
                     font=("Segoe UI", 14, "bold"), relief="flat")
    entry.pack(padx=14, pady=(0, 12), ipadx=10, ipady=6)

    notify = tk.Label(win, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 11))
    notify.pack(pady=(6, 6))

    # ---------------- STATE FLAG ----------------
    state = {"running": False}

    # ---------------- SAFE TTS ----------------
    def safe_tts(msg):
        try:
            text_to_speech(msg)
        except:
            pass

    # ===================================================================
    #                          SHOW CSV
    # ===================================================================
    def show_csv(path, title):
        df = pd.read_csv(path)

        tv = tk.Toplevel(win)
        tv.title(title)
        tv.configure(bg=BG)

        tree = ttk.Treeview(tv, show="headings")
        tree.pack(fill="both", expand=True)

        cols = list(df.columns)
        tree["columns"] = cols

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=150, anchor="center")

        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        tk.Button(tv, text="Close", command=tv.destroy,
                  bg=BTN_BG, fg=ACCENT).pack(pady=6)

    # ===================================================================
    #                        OPEN FOLDER
    # ===================================================================

    def open_folder():
        sub = entry.get().strip()
        if not sub:
            notify.config(text="⚠ Enter subject name")
            safe_tts("Please enter subject name")
            return

        folder = os.path.join(attendance_path, sub)
        os.makedirs(folder, exist_ok=True)

        try:
            os.startfile(folder)
        except:
            messagebox.showinfo("Folder Path", folder)

    # ===================================================================
    #                   ATTENDANCE FUNCTION (NO THREAD)
    # ===================================================================

    def attendance_thread(subject):

        state["running"] = True

        notify.config(text="Starting camera…")
        safe_tts(f"Starting attendance for {subject}")

        # -------- Load Recognizer --------
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except:
            notify.config(text="Install opencv-contrib-python!")
            safe_tts("OpenCV face module missing")
            state["running"] = False
            return

        # -------- Load Trained Model --------
        if not os.path.exists(TRAIN_LABEL_PATH):
            notify.config(text="Train the model first!")
            safe_tts("Training file missing")
            state["running"] = False
            return

        recognizer.read(TRAIN_LABEL_PATH)

        # -------- Load Haar Cascade --------
        if not os.path.exists(HAAR_PATH):
            notify.config(text="Haarcascade file missing")
            safe_tts("Haarcascade missing")
            state["running"] = False
            return

        face_cascade = cv2.CascadeClassifier(HAAR_PATH)

        # -------- Load Student CSV --------
        if not os.path.exists(studentdetail_path):
            notify.config(text="Student details not found")
            safe_tts("Student details missing")
            state["running"] = False
            return

        df_students = pd.read_csv(studentdetail_path, dtype={"Enrollment": str})

        # -------- Start Camera --------
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            notify.config(text="Camera not detected!")
            safe_tts("Camera not found")
            state["running"] = False
            return

        found = {}
        timeout = time.time() + 20  # 20 seconds

        # ================= CAMERA LOOP =================
        while time.time() < timeout:

            ret, frame = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]

                try:
                    Id, conf = recognizer.predict(roi)
                except:
                    continue

                if conf < 70:
                    Id = str(Id)
                    stu = df_students[df_students["Enrollment"] == Id]

                    name = stu["Name"].values[0] if not stu.empty else "Unknown"
                    found[Id] = name

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{Id}-{name}", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Unknown", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("CLASS VISION - Attendance (ESC to exit)", frame)

            # ESC to exit early
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        # ============================================================
        #                     SAVE ATTENDANCE
        # ============================================================
        if not found:
            notify.config(text="No face recognized!")
            safe_tts("No face recognized")
            state["running"] = False
            return

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        tstamp = datetime.datetime.now().strftime("%H-%M-%S")

        rows = [
            {"Enrollment": k, "Name": v, "Date": today, "Present": 1}
            for k, v in found.items()
        ]

        df_att = pd.DataFrame(rows)

        folder = os.path.join(attendance_path, subject)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{subject}_{today}_{tstamp}.csv")

        df_att.to_csv(filepath, index=False)

        notify.config(text="Attendance saved!")
        safe_tts("Attendance saved successfully")

        show_csv(filepath, f"Attendance – {subject}")

        state["running"] = False

    # ===================================================================
    #                         BUTTONS
    # ===================================================================

    def start():
        sub = entry.get().strip()

        if not sub:
            notify.config(text="⚠ Enter subject name")
            safe_tts("Please enter subject name")
            return

        if state["running"]:
            notify.config(text="Attendance already running…")
            return

        attendance_thread(sub)  # MAIN THREAD ONLY (mac safe)

    # ---------------- BUTTON FRAME ----------------
    btn_frame = tk.Frame(win, bg=BG)
    btn_frame.pack()

    tk.Button(btn_frame, text="Fill Attendance", command=start,
              bg=BTN_BG, fg=ACCENT, bd=0,
              padx=16, pady=8).grid(row=0, column=0, padx=8)

    tk.Button(btn_frame, text="Open Folder", command=open_folder,
              bg=BTN_BG, fg=ACCENT, bd=0,
              padx=16, pady=8).grid(row=0, column=1, padx=8)

    tk.Button(win, text="Close", command=win.destroy,
              bg="#222222", fg=ACCENT, bd=0,
              padx=16, pady=8).pack(pady=12)

    entry.focus_set()
    win.grab_set()
