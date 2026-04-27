# ============================================================
# desktop_app.py — CustomTkinter Desktop Application
# ============================================================
# HOW TO RUN:
#   Step 1: pip install customtkinter torch mrcfile matplotlib pillow
#   Step 2: python desktop_app.py
#
# HOW TO MAKE .EXE:
#   pip install pyinstaller
#   pyinstaller --onefile --windowed --name ProteinID desktop_app.py
# ============================================================

import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mrcfile
import os
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import io

# ── App Theme ────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Constants ────────────────────────────────────────────
VOLUME_SIZE = 64
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_MAP = {
    0: 'Ribosome (80S)',       1: 'Ribosome (55S)',
    2: 'Proteasome',           3: 'Proteasome (Human)',
    4: 'ATP-Synthase',         5: 'GroEL',
    6: 'Virus-Like Particle',  7: 'TRiC/CCT',
    8: 'Cytoplasmic Dynein',   9: 'Fatty Acid Synthase',
}

CLASS_COLORS = [
    '#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444',
    '#ec4899','#06b6d4','#84cc16','#f97316','#8b5cf6'
]

CLASS_INFO = {
    0: 'Large protein-RNA complex. Responsible for protein synthesis in all living cells.',
    1: 'Mitochondrial ribosome. Synthesizes proteins inside mitochondria.',
    2: 'Barrel-shaped complex. Degrades damaged/unneeded proteins.',
    3: 'Human proteasome variant. Involved in immune response.',
    4: 'Produces ATP energy. Found in mitochondrial membrane.',
    5: 'Chaperone protein. Helps other proteins fold correctly.',
    6: 'Hollow protein shell. Used in vaccine research.',
    7: 'Eukaryotic chaperonin. Assists protein folding.',
    8: 'Motor protein complex. Transports cellular cargo.',
    9: 'Metabolic enzyme. Synthesizes fatty acids.',
}


# ── 3D CNN Model ─────────────────────────────────────────
class Protein3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv3d(1,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2))
        self.block2 = nn.Sequential(nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2))
        self.block3 = nn.Sequential(nn.Conv3d(64,128,3,padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2))
        self.block4 = nn.Sequential(nn.Conv3d(128,256,3,padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.gap    = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.classifier(self.gap(x))


# ── Helper Functions ──────────────────────────────────────
def preprocess_volume(filepath):
    with mrcfile.open(filepath, permissive=True) as mrc:
        vol = mrc.data.astype(np.float32)
    if vol.ndim != 3:
        raise ValueError("Invalid volume - not 3D!")
    d, h, w = vol.shape
    ds = max((d-VOLUME_SIZE)//2, 0)
    hs = max((h-VOLUME_SIZE)//2, 0)
    ws = max((w-VOLUME_SIZE)//2, 0)
    vol = vol[ds:ds+VOLUME_SIZE, hs:hs+VOLUME_SIZE, ws:ws+VOLUME_SIZE]
    result = np.zeros((VOLUME_SIZE,)*3, dtype=np.float32)
    s = vol.shape
    result[:s[0],:s[1],:s[2]] = vol[:s[0],:s[1],:s[2]]
    vmin, vmax = result.min(), result.max()
    if vmax - vmin > 1e-8:
        result = (result - vmin)/(vmax - vmin)
    return result


# ════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════
class ProteinApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("🧬 ProteinID — Cryo-ET Protein Complex Identifier")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.configure(fg_color="#080b14")

        self.model       = None
        self.volume      = None
        self.selected_file = None

        self._build_ui()
        self._load_model()

    # ── BUILD UI ──────────────────────────────────────────
    def _build_ui(self):

        # ── TOP NAVBAR ──
        self.navbar = ctk.CTkFrame(self, fg_color="#0e1525", height=60, corner_radius=0)
        self.navbar.pack(fill="x", pady=(0,0))
        self.navbar.pack_propagate(False)

        ctk.CTkLabel(
            self.navbar, text="🧬  ProteinID",
            font=ctk.CTkFont("Helvetica", 20, "bold"),
            text_color="#00d4ff"
        ).pack(side="left", padx=24, pady=15)

        ctk.CTkLabel(
            self.navbar, text="Automated Protein Complex Identification · SHREC 2019 · 3D-CNN",
            font=ctk.CTkFont("Helvetica", 11),
            text_color="#64748b"
        ).pack(side="left", padx=0)

        self.model_status_label = ctk.CTkLabel(
            self.navbar, text="⚪  Loading model...",
            font=ctk.CTkFont("Helvetica", 11),
            text_color="#64748b"
        )
        self.model_status_label.pack(side="right", padx=24)

        # ── MAIN CONTENT ──
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=16, pady=12)

        # Left panel (fixed width)
        self.left_panel = ctk.CTkFrame(self.content, fg_color="transparent", width=340)
        self.left_panel.pack(side="left", fill="y", padx=(0,12))
        self.left_panel.pack_propagate(False)

        # Right panel
        self.right_panel = ctk.CTkFrame(self.content, fg_color="transparent")
        self.right_panel.pack(side="left", fill="both", expand=True)

        self._build_left_panel()
        self._build_right_panel()

    # ── LEFT PANEL ────────────────────────────────────────
    def _build_left_panel(self):

        # Upload Card
        upload_card = ctk.CTkFrame(self.left_panel, fg_color="#0e1525", corner_radius=12,
                                    border_width=1, border_color="#1e2d4a")
        upload_card.pack(fill="x", pady=(0,12))

        ctk.CTkLabel(upload_card, text="📤  Upload .mrc File",
                     font=ctk.CTkFont("Helvetica", 13, "bold"),
                     text_color="#00d4ff").pack(anchor="w", padx=18, pady=(16,10))

        # Upload button area
        self.upload_frame = ctk.CTkFrame(upload_card, fg_color="#131c30",
                                          corner_radius=10, height=120)
        self.upload_frame.pack(fill="x", padx=14, pady=(0,12))
        self.upload_frame.pack_propagate(False)

        ctk.CTkLabel(self.upload_frame, text="🔬",
                     font=ctk.CTkFont("Helvetica", 36)).pack(pady=(14,4))
        ctk.CTkLabel(self.upload_frame, text="Click to select .mrc file",
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color="#94a3b8").pack()

        self.file_label = ctk.CTkLabel(upload_card, text="No file selected",
                                        font=ctk.CTkFont("Helvetica", 11),
                                        text_color="#64748b")
        self.file_label.pack(padx=14, pady=(0,8))

        ctk.CTkButton(
            upload_card, text="📂  Browse File",
            font=ctk.CTkFont("Helvetica", 12, "bold"),
            fg_color="#1e2d4a", hover_color="#2d3f5a",
            text_color="#00d4ff", height=36,
            command=self._browse_file
        ).pack(fill="x", padx=14, pady=(0,8))

        self.predict_btn = ctk.CTkButton(
            upload_card, text="🧬  Identify Protein",
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            fg_color="#00d4ff", hover_color="#00b8e0",
            text_color="#080b14", height=42,
            command=self._start_prediction,
            state="disabled"
        )
        self.predict_btn.pack(fill="x", padx=14, pady=(0,16))

        # Progress
        self.progress_label = ctk.CTkLabel(upload_card, text="",
                                            font=ctk.CTkFont("Helvetica", 10),
                                            text_color="#64748b")
        self.progress_label.pack(pady=(0,8))

        self.progress_bar = ctk.CTkProgressBar(upload_card, height=4,
                                                progress_color="#00d4ff",
                                                fg_color="#1e2d4a")
        self.progress_bar.pack(fill="x", padx=14, pady=(0,14))
        self.progress_bar.set(0)

        # Model Info Card
        info_card = ctk.CTkFrame(self.left_panel, fg_color="#0e1525", corner_radius=12,
                                  border_width=1, border_color="#1e2d4a")
        info_card.pack(fill="x", pady=(0,12))

        ctk.CTkLabel(info_card, text="🧠  Model Info",
                     font=ctk.CTkFont("Helvetica", 13, "bold"),
                     text_color="#00d4ff").pack(anchor="w", padx=18, pady=(14,10))

        info_items = [
            ("Architecture", "3D-CNN (4 Blocks)"),
            ("Parameters",   "3,842,314"),
            ("Volume Size",  "64 × 64 × 64"),
            ("Classes",      "10 Protein Types"),
            ("Dataset",      "SHREC 2019"),
            ("Framework",    "PyTorch"),
        ]
        for lbl, val in info_items:
            row = ctk.CTkFrame(info_card, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(row, text=lbl+":", font=ctk.CTkFont("Helvetica", 11),
                         text_color="#64748b", width=110, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=val, font=ctk.CTkFont("Helvetica", 11, "bold"),
                         text_color="#e2e8f0").pack(side="left")

        ctk.CTkFrame(info_card, height=12, fg_color="transparent").pack()

        # Classes Card
        cls_card = ctk.CTkFrame(self.left_panel, fg_color="#0e1525", corner_radius=12,
                                 border_width=1, border_color="#1e2d4a")
        cls_card.pack(fill="both", expand=True)

        ctk.CTkLabel(cls_card, text="🏷️  Protein Classes",
                     font=ctk.CTkFont("Helvetica", 13, "bold"),
                     text_color="#00d4ff").pack(anchor="w", padx=18, pady=(14,8))

        scroll = ctk.CTkScrollableFrame(cls_card, fg_color="transparent", height=180)
        scroll.pack(fill="both", expand=True, padx=10, pady=(0,10))

        for i, name in CLASS_MAP.items():
            row = ctk.CTkFrame(scroll, fg_color="#131c30", corner_radius=7,
                               border_width=1, border_color="#1e2d4a")
            row.pack(fill="x", pady=2, padx=2)
            ctk.CTkLabel(row, text="●", font=ctk.CTkFont("Helvetica", 10),
                         text_color=CLASS_COLORS[i], width=20).pack(side="left", padx=(8,4), pady=6)
            ctk.CTkLabel(row, text=f"{i}  {name}",
                         font=ctk.CTkFont("Helvetica", 11),
                         text_color="#94a3b8").pack(side="left", pady=6)

    # ── RIGHT PANEL ───────────────────────────────────────
    def _build_right_panel(self):

        # Result area (initially shows welcome)
        self.result_frame = ctk.CTkFrame(self.right_panel, fg_color="#0e1525",
                                          corner_radius=12, border_width=1,
                                          border_color="#1e2d4a")
        self.result_frame.pack(fill="both", expand=True)

        self._show_welcome()

    def _show_welcome(self):
        for w in self.result_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.result_frame, text="🧬",
                     font=ctk.CTkFont("Helvetica", 72)).pack(pady=(80,10))
        ctk.CTkLabel(self.result_frame,
                     text="Welcome to ProteinID",
                     font=ctk.CTkFont("Helvetica", 24, "bold"),
                     text_color="#e2e8f0").pack()
        ctk.CTkLabel(self.result_frame,
                     text="Upload a .mrc file and click 'Identify Protein'\nto classify your Cryo-ET tomogram",
                     font=ctk.CTkFont("Helvetica", 13),
                     text_color="#64748b", justify="center").pack(pady=14)

        # Feature tags
        tags_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        tags_frame.pack(pady=10)
        for tag in ["3D Volume Analysis", "10 Protein Classes", "Deep Learning", "SHREC 2019"]:
            ctk.CTkLabel(tags_frame, text=f"  {tag}  ",
                         font=ctk.CTkFont("Helvetica", 11),
                         fg_color="#131c30", text_color="#00d4ff",
                         corner_radius=8).pack(side="left", padx=5)

    # ── FILE BROWSE ───────────────────────────────────────
    def _browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select .mrc File",
            filetypes=[("MRC Files", "*.mrc"), ("All Files", "*.*")]
        )
        if filepath:
            self.selected_file = filepath
            fname = os.path.basename(filepath)
            fsize = os.path.getsize(filepath) // 1024
            self.file_label.configure(
                text=f"📂 {fname[:30]}...\n{fsize:,} KB",
                text_color="#10b981"
            )
            self.predict_btn.configure(state="normal")

    # ── LOAD MODEL ────────────────────────────────────────
    def _load_model(self):
        def load():
            try:
                if os.path.exists('protein_model.pth'):
                    self.model = Protein3DCNN(num_classes=10).to(DEVICE)
                    self.model.load_state_dict(
                        torch.load('protein_model.pth', map_location=DEVICE)
                    )
                    self.model.eval()
                    self.after(0, lambda: self.model_status_label.configure(
                        text="🟢  Model Ready", text_color="#10b981"
                    ))
                else:
                    self.after(0, lambda: self.model_status_label.configure(
                        text="🔴  Model Not Found", text_color="#ef4444"
                    ))
            except Exception as e:
                self.after(0, lambda: self.model_status_label.configure(
                    text=f"🔴  Error: {str(e)[:30]}", text_color="#ef4444"
                ))
        threading.Thread(target=load, daemon=True).start()

    # ── PREDICTION ────────────────────────────────────────
    def _start_prediction(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a .mrc file first!")
            return
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded! Run notebook first.")
            return

        self.predict_btn.configure(state="disabled", text="⏳  Analyzing...")
        self.progress_bar.set(0)
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            steps = [
                (0.2, "📂 Loading .mrc file..."),
                (0.4, "✂️  Cropping center..."),
                (0.6, "🔧 Normalizing volume..."),
                (0.8, "🧠 Running 3D-CNN..."),
                (1.0, "📊 Generating results..."),
            ]

            for prog, msg in steps:
                self.after(0, lambda p=prog, m=msg: (
                    self.progress_bar.set(p),
                    self.progress_label.configure(text=m)
                ))
                import time; time.sleep(0.4)

            # Preprocess
            self.volume = preprocess_volume(self.selected_file)

            # Predict
            tensor = torch.tensor(self.volume).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output     = self.model(tensor)
                probs      = F.softmax(output, dim=1)[0].cpu().numpy()
                pred_class = int(probs.argmax())
                confidence = float(probs[pred_class]) * 100

            self.after(0, lambda pc=pred_class, cf=confidence, pb=probs: self._show_results(pc, cf, pb))

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: (
                messagebox.showerror("Prediction Error", msg),
                self.predict_btn.configure(state="normal", text="🧬  Identify Protein"),
                self.progress_label.configure(text=""),
                self.progress_bar.set(0)
            ))

    # ── SHOW RESULTS ──────────────────────────────────────
    def _show_results(self, pred_class, confidence, probs):
        # Reset UI
        self.predict_btn.configure(state="normal", text="🧬  Identify Protein")
        self.progress_label.configure(text="✅ Done!")
        self.progress_bar.set(1.0)

        pred_name  = CLASS_MAP[pred_class]
        pred_color = CLASS_COLORS[pred_class]
        pred_info  = CLASS_INFO[pred_class]

        # Clear right panel
        for w in self.result_frame.winfo_children():
            w.destroy()

        # ── RESULT HEADER ──
        header = ctk.CTkFrame(self.result_frame, fg_color="#131c30",
                               corner_radius=10, border_width=2,
                               border_color=pred_color)
        header.pack(fill="x", padx=14, pady=(14,10))

        left_h = ctk.CTkFrame(header, fg_color="transparent")
        left_h.pack(side="left", fill="both", expand=True, padx=18, pady=14)

        ctk.CTkLabel(left_h, text="🎯  PREDICTED PROTEIN COMPLEX",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color="#64748b").pack(anchor="w")
        ctk.CTkLabel(left_h, text=pred_name,
                     font=ctk.CTkFont("Helvetica", 24, "bold"),
                     text_color=pred_color).pack(anchor="w", pady=(4,6))
        ctk.CTkLabel(left_h, text=pred_info,
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color="#94a3b8", wraplength=400, justify="left").pack(anchor="w")

        conf_box = ctk.CTkFrame(header, fg_color="#080b14", corner_radius=10, width=130)
        conf_box.pack(side="right", padx=18, pady=14)
        conf_box.pack_propagate(False)

        conf_color = "#10b981" if confidence > 85 else "#f59e0b"
        ctk.CTkLabel(conf_box, text=f"{confidence:.1f}%",
                     font=ctk.CTkFont("Helvetica", 32, "bold"),
                     text_color=conf_color).pack(pady=(16,2))
        ctk.CTkLabel(conf_box, text="Confidence",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color="#64748b").pack(pady=(0,16))

        # ── TABS ──
        tab_view = ctk.CTkTabview(self.result_frame, fg_color="#0e1525",
                                   segmented_button_fg_color="#131c30",
                                   segmented_button_selected_color="#00d4ff",
                                   segmented_button_selected_hover_color="#00b8e0",
                                   segmented_button_unselected_color="#131c30",
                                   text_color="#e2e8f0")
        tab_view.pack(fill="both", expand=True, padx=14, pady=(0,14))

        tab1 = tab_view.add("🔍  3D Slices")
        tab2 = tab_view.add("📊  Confidence")
        tab3 = tab_view.add("📈  All Classes")

        # ── TAB 1: 3D Slices ──
        self._build_slices_tab(tab1, pred_color)

        # ── TAB 2: Confidence Bars ──
        self._build_confidence_tab(tab2, probs, pred_class)

        # ── TAB 3: All Classes Chart ──
        self._build_chart_tab(tab3, probs, pred_class)

    # ── 3D SLICES TAB ─────────────────────────────────────
    def _build_slices_tab(self, parent, color):
        if self.volume is None:
            return

        mid  = VOLUME_SIZE // 2
        cmap = LinearSegmentedColormap.from_list('v', ['#0a0a1a','#1a3a6a', color, 'white'])

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        fig.patch.set_facecolor('#0e1525')

        slices = [
            (self.volume[:,:,mid], 'XY Slice (Top)'),
            (self.volume[:,mid,:], 'XZ Slice (Front)'),
            (self.volume[mid,:,:], 'YZ Slice (Side)'),
        ]

        for ax, (slc, title) in zip(axes, slices):
            ax.imshow(slc, cmap=cmap, origin='lower')
            ax.set_title(title, color=color, fontsize=10,
                         fontweight='bold', fontfamily='monospace')
            ax.set_facecolor('#131c30')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)

        plt.tight_layout(pad=1.0)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        plt.close(fig)

    # ── CONFIDENCE BARS TAB ───────────────────────────────
    def _build_confidence_tab(self, parent, probs, pred_class):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=8, pady=8)

        sorted_idx = np.argsort(probs)[::-1]
        for i in sorted_idx:
            val    = probs[i] * 100
            color  = CLASS_COLORS[i]
            is_pred = (i == pred_class)

            row = ctk.CTkFrame(scroll, fg_color="#131c30" if is_pred else "#0e1525",
                               corner_radius=8,
                               border_width=1 if is_pred else 0,
                               border_color=color if is_pred else "#1e2d4a")
            row.pack(fill="x", pady=3, padx=2)

            ctk.CTkLabel(row, text=CLASS_MAP[i],
                         font=ctk.CTkFont("Helvetica", 11, "bold" if is_pred else "normal"),
                         text_color=color if is_pred else "#94a3b8",
                         width=180, anchor="w").pack(side="left", padx=(12,8), pady=8)

            bar_frame = ctk.CTkFrame(row, fg_color="#1e2d4a", corner_radius=4, height=12)
            bar_frame.pack(side="left", fill="x", expand=True, pady=8)

            bar_fill = ctk.CTkFrame(bar_frame, fg_color=color if is_pred else "#2d3748",
                                     corner_radius=4, height=12,
                                     width=int(val * 3))
            bar_fill.place(x=0, y=0, relheight=1)

            ctk.CTkLabel(row, text=f"{val:.1f}%",
                         font=ctk.CTkFont("JetBrains Mono" if True else "Helvetica", 11,
                                          "bold" if is_pred else "normal"),
                         text_color=color if is_pred else "#64748b",
                         width=55).pack(side="right", padx=10)

    # ── ALL CLASSES CHART TAB ─────────────────────────────
    def _build_chart_tab(self, parent, probs, pred_class):
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0e1525')
        ax.set_facecolor('#131c30')

        labels = [CLASS_MAP[i] for i in range(10)]
        colors = [CLASS_COLORS[i] if i == pred_class else '#2d3748' for i in range(10)]
        vals   = [p*100 for p in probs]

        bars = ax.bar(range(10), vals, color=colors, edgecolor='#1e2d4a', linewidth=0.5)

        for bar, val, i in zip(bars, vals, range(10)):
            if val > 1:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{val:.1f}%', ha='center', fontsize=8,
                        color=CLASS_COLORS[i] if i==pred_class else '#64748b',
                        fontweight='bold')

        ax.set_xticks(range(10))
        ax.set_xticklabels([n.split('(')[0].strip()[:10] for n in labels],
                            rotation=25, ha='right', color='#94a3b8', fontsize=9)
        ax.set_ylabel('Confidence (%)', color='#94a3b8', fontsize=10)
        ax.set_title('Prediction Confidence — All Classes',
                     color='#e2e8f0', fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors='#64748b')
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', linestyle='--', alpha=0.2, color='#2d3748')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d4a')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        plt.close(fig)


# ════════════════════════════════════════════════════════
# RUN APP
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ProteinApp()
    app.mainloop()