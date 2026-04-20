# -*- coding: utf-8 -*- 
"""
YOLOv11 行人与车辆检测可视化程序
使用方法：python detection_app.py
依赖：pip install ultralytics opencv-python pillow tkinter
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import time
import os
import cv2
import json
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import defaultdict

# ── 颜色配置 ──────────────────────────────────────────────────
COLORS = {
    'bg':        '#0F1117',
    'panel':     '#1A1D27',
    'card':      '#22263A',
    'accent':    '#4F8EF7',
    'accent2':   '#7C3AED',
    'success':   '#10B981',
    'warning':   '#F59E0B',
    'danger':    '#EF4444',
    'text':      '#F1F5F9',
    'subtext':   '#94A3B8',
    'border':    '#2E3350',
}

# 每个类别对应的检测框颜色（BGR格式，用于OpenCV）
CLASS_COLORS_BGR = {
    'person':     (255, 100,  50),
    'bicycle':    ( 50, 255, 150),
    'car':        ( 50, 150, 255),
    'motorcycle': (255, 200,  50),
    'bus':        (200,  50, 255),
    'truck':      ( 50, 255, 255),
}


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 行人与车辆检测系统")
        self.root.geometry("1200x780")
        self.root.configure(bg=COLORS['bg'])
        self.root.resizable(True, True)

        # 状态变量
        self.model = None
        self.model_path = tk.StringVar(value="yolo11m.pt")
        self.is_running = False
        self.is_paused = False
        self.current_file = None
        self.file_type = None   # 'image' or 'video'
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.result_stats = defaultdict(int)
        self.fps_var = tk.StringVar(value="0")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(value="请先加载模型，再上传图片或视频")

        self._build_ui()

    # ══════════════════════════════════════════════════════════
    #  UI 构建
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        # 顶部标题栏
        self._build_topbar()
        # 主体区域
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=16, pady=(0, 12))
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        # 左侧预览区
        self._build_preview(main)
        # 右侧控制面板
        self._build_sidebar(main)
        # 底部状态栏
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=COLORS['panel'], height=56)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        tk.Label(bar, text="🚗  YOLOv11", font=('Helvetica', 18, 'bold'),
                 bg=COLORS['panel'], fg=COLORS['accent']).pack(side='left', padx=20, pady=10)
        tk.Label(bar, text="行人与车辆检测系统",
                 font=('Helvetica', 13), bg=COLORS['panel'], fg=COLORS['text']).pack(side='left', pady=10)

        # FPS 显示
        fps_frame = tk.Frame(bar, bg=COLORS['card'], padx=10, pady=4)
        fps_frame.pack(side='right', padx=20, pady=10)
        tk.Label(fps_frame, text="FPS", font=('Helvetica', 9),
                 bg=COLORS['card'], fg=COLORS['subtext']).pack(side='left')
        tk.Label(fps_frame, textvariable=self.fps_var, font=('Helvetica', 14, 'bold'),
                 bg=COLORS['card'], fg=COLORS['success'], width=4).pack(side='left', padx=(4, 0))

    def _build_preview(self, parent):
        frame = tk.Frame(parent, bg=COLORS['bg'])
        frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # 画布容器
        canvas_bg = tk.Frame(frame, bg=COLORS['panel'], bd=0,
                              highlightthickness=1, highlightbackground=COLORS['border'])
        canvas_bg.grid(row=0, column=0, sticky='nsew')
        canvas_bg.rowconfigure(0, weight=1)
        canvas_bg.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_bg, bg=COLORS['panel'],
                                bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # 占位文字
        self.placeholder_id = self.canvas.create_text(
            400, 300,
            text="📂  点击右侧按钮上传图片或视频",
            font=('Helvetica', 16), fill=COLORS['subtext']
        )

        # 进度条（视频用）
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Scale(frame, from_=0, to=100,
                                  orient='horizontal', variable=self.progress_var,
                                  command=self._on_seek)
        self.progress.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Horizontal.TScale', background=COLORS['bg'],
                        troughcolor=COLORS['card'], slidercolor=COLORS['accent'])

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLORS['bg'], width=280)
        sidebar.grid(row=0, column=1, sticky='nsew')
        sidebar.pack_propagate(False)

        # ── 模型加载卡片 ──
        self._card(sidebar, "⚙️  模型设置", self._build_model_card)
        # ── 文件操作卡片 ──
        self._card(sidebar, "📂  文件操作", self._build_file_card)
        # ── 参数设置卡片 ──
        self._card(sidebar, "🎛️  检测参数", self._build_param_card)
        # ── 统计结果卡片 ──
        self._card(sidebar, "📊  检测统计", self._build_stats_card)

    def _card(self, parent, title, builder_fn):
        card = tk.Frame(parent, bg=COLORS['card'],
                        highlightthickness=1, highlightbackground=COLORS['border'])
        card.pack(fill='x', pady=(0, 10))

        header = tk.Frame(card, bg=COLORS['panel'])
        header.pack(fill='x')
        tk.Label(header, text=title, font=('Helvetica', 10, 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text'], anchor='w',
                 padx=12, pady=8).pack(fill='x')

        body = tk.Frame(card, bg=COLORS['card'], padx=12, pady=10)
        body.pack(fill='x')
        builder_fn(body)

    def _build_model_card(self, parent):
        tk.Label(parent, text="权重文件路径", font=('Helvetica', 9),
                 bg=COLORS['card'], fg=COLORS['subtext'], anchor='w').pack(fill='x')

        row = tk.Frame(parent, bg=COLORS['card'])
        row.pack(fill='x', pady=(4, 8))
        entry = tk.Entry(row, textvariable=self.model_path,
                         bg=COLORS['bg'], fg=COLORS['text'],
                         insertbackground=COLORS['text'], bd=0,
                         font=('Helvetica', 9), relief='flat')
        entry.pack(side='left', fill='x', expand=True, ipady=5, padx=(0, 6))

        self._btn(row, "浏览", self._browse_model, width=6).pack(side='right')

        self.load_btn = self._btn(parent, "🔄  加载模型", self._load_model,
                                  color=COLORS['accent'])
        self.load_btn.pack(fill='x')

        self.model_status = tk.Label(parent, text="● 未加载",
                                     font=('Helvetica', 9), bg=COLORS['card'],
                                     fg=COLORS['danger'], anchor='w')
        self.model_status.pack(fill='x', pady=(6, 0))

    def _build_file_card(self, parent):
        self._btn(parent, "🖼️  上传图片",
                  lambda: self._open_file('image'),
                  color=COLORS['success']).pack(fill='x', pady=(0, 6))
        self._btn(parent, "🎬  上传视频",
                  lambda: self._open_file('video'),
                  color=COLORS['accent2']).pack(fill='x', pady=(0, 6))

        ctrl = tk.Frame(parent, bg=COLORS['card'])
        ctrl.pack(fill='x')
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)

        self.play_btn = self._btn(ctrl, "▶  开始", self._toggle_play,
                                  color=COLORS['accent'])
        self.play_btn.grid(row=0, column=0, sticky='ew', padx=(0, 4))

        self._btn(ctrl, "⏹  停止", self._stop,
                  color=COLORS['danger']).grid(row=0, column=1, sticky='ew')

        self._btn(parent, "💾  保存结果", self._save_result,
                  color=COLORS['warning']).pack(fill='x', pady=(6, 0))

    def _build_param_card(self, parent):
        tk.Label(parent, text=f"置信度阈值: {self.conf_var.get():.2f}",
                 font=('Helvetica', 9), bg=COLORS['card'],
                 fg=COLORS['subtext'], anchor='w').pack(fill='x')

        self.conf_label_var = tk.StringVar(value="0.25")

        def on_conf(v):
            val = float(v)
            self.conf_label_var.set(f"{val:.2f}")
            conf_display.config(text=f"置信度阈值: {val:.2f}")

        conf_display = tk.Label(parent, text="置信度阈值: 0.25",
                                font=('Helvetica', 9), bg=COLORS['card'],
                                fg=COLORS['subtext'], anchor='w')
        conf_display.pack(fill='x')

        ttk.Scale(parent, from_=0.05, to=0.95, orient='horizontal',
                  variable=self.conf_var, command=on_conf).pack(fill='x', pady=(2, 8))

    def _build_stats_card(self, parent):
        self.stats_frame = tk.Frame(parent, bg=COLORS['card'])
        self.stats_frame.pack(fill='x')
        self._refresh_stats()

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=COLORS['panel'], height=30)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_var, font=('Helvetica', 9),
                 bg=COLORS['panel'], fg=COLORS['subtext'], anchor='w',
                 padx=16).pack(fill='x', pady=6)

    def _btn(self, parent, text, command, color=None, width=None):
        color = color or COLORS['card']
        b = tk.Button(parent, text=text, command=command,
                      bg=color, fg=COLORS['text'],
                      font=('Helvetica', 9, 'bold'),
                      bd=0, relief='flat', cursor='hand2',
                      activebackground=color, activeforeground=COLORS['text'],
                      padx=8, pady=6)
        if width:
            b.config(width=width)
        # 悬停效果
        def on_enter(e): b.config(bg=self._lighten(color))
        def on_leave(e): b.config(bg=color)
        b.bind('<Enter>', on_enter)
        b.bind('<Leave>', on_leave)
        return b

    def _lighten(self, hex_color):
        """让颜色稍微亮一点，用于 hover 效果"""
        try:
            r = min(255, int(hex_color[1:3], 16) + 30)
            g = min(255, int(hex_color[3:5], 16) + 30)
            b = min(255, int(hex_color[5:7], 16) + 30)
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return hex_color

    # ══════════════════════════════════════════════════════════
    #  模型加载
    # ══════════════════════════════════════════════════════════
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch权重", "*.pt"), ("所有文件", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def _load_model(self):
        self.model_status.config(text="● 加载中...", fg=COLORS['warning'])
        self.status_var.set("正在加载模型，请稍候...")
        self.root.update()

        def _do_load():
            try:
                self.model = YOLO(self.model_path.get())
                self.root.after(0, lambda: self.model_status.config(
                    text=f"● 已加载：{os.path.basename(self.model_path.get())}",
                    fg=COLORS['success']))
                self.root.after(0, lambda: self.status_var.set(
                    "✅ 模型加载成功！请上传图片或视频开始检测"))
            except Exception as e:
                self.root.after(0, lambda: self.model_status.config(
                    text=f"● 加载失败", fg=COLORS['danger']))
                self.root.after(0, lambda: self.status_var.set(f"❌ 加载失败：{e}"))

        threading.Thread(target=_do_load, daemon=True).start()

    # ══════════════════════════════════════════════════════════
    #  文件操作
    # ══════════════════════════════════════════════════════════
    def _open_file(self, ftype):
        if not self.model:
            self.status_var.set("⚠️ 请先加载模型！")
            return

        self._stop()

        if ftype == 'image':
            path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
            )
        else:
            path = filedialog.askopenfilename(
                title="选择视频",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"), ("所有文件", "*.*")]
            )

        if not path:
            return

        self.current_file = path
        self.file_type = ftype
        self.result_stats.clear()

        if ftype == 'image':
            self.status_var.set(f"已加载图片：{os.path.basename(path)}")
            self._run_image()
        else:
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_var.set(0)
            self.status_var.set(f"已加载视频：{os.path.basename(path)}  共 {self.total_frames} 帧")
            # 显示第一帧预览
            ret, frame = self.cap.read()
            if ret:
                self._show_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _toggle_play(self):
        if self.file_type == 'image':
            self._run_image()
            return

        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.play_btn.config(text="⏸  暂停")
            threading.Thread(target=self._run_video, daemon=True).start()
        else:
            self.is_paused = not self.is_paused
            self.play_btn.config(text="▶  继续" if self.is_paused else "⏸  暂停")

    def _stop(self):
        self.is_running = False
        self.is_paused = False
        self.play_btn.config(text="▶  开始")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.progress_var.set(0)

    def _on_seek(self, val):
        if self.cap and self.file_type == 'video':
            frame_idx = int(float(val) / 100 * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # ══════════════════════════════════════════════════════════
    #  检测逻辑
    # ══════════════════════════════════════════════════════════
    def _run_image(self):
        """图片单帧检测"""
        if not self.current_file or not self.model:
            return

        self.status_var.set("🔍 正在检测中...")
        self.root.update()

        def _do():
            frame = cv2.imread(self.current_file)
            t0 = time.time()
            results = self.model(frame, conf=self.conf_var.get(), verbose=False)
            fps = 1.0 / max(time.time() - t0, 1e-6)

            annotated, stats = self._annotate(frame, results[0])
            self.root.after(0, lambda: self._show_frame(annotated))
            self.root.after(0, lambda: self.fps_var.set(f"{fps:.0f}"))
            self.result_stats = stats
            self.root.after(0, self._refresh_stats)

            total = sum(stats.values())
            self.root.after(0, lambda: self.status_var.set(
                f"✅ 检测完成  共发现 {total} 个目标  " +
                "  ".join(f"{k}:{v}" for k, v in stats.items() if v > 0)
            ))
            self._last_annotated = annotated

        threading.Thread(target=_do, daemon=True).start()

    def _run_video(self):
        """视频逐帧检测"""
        if not self.cap or not self.model:
            return

        fps_times = []
        self._last_annotated = None

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.status_var.set("✅ 视频播放完毕"))
                break

            # 检测
            results = self.model(frame, conf=self.conf_var.get(), verbose=False)
            annotated, stats = self._annotate(frame, results[0])
            self._last_annotated = annotated

            # 累计统计
            for k, v in stats.items():
                self.result_stats[k] += v

            # FPS
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 20:
                fps_times.pop(0)
            avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

            # 进度
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = pos / max(self.total_frames, 1) * 100

            self.root.after(0, lambda f=annotated: self._show_frame(f))
            self.root.after(0, lambda f=avg_fps: self.fps_var.set(f"{f:.0f}"))
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
            self.root.after(0, self._refresh_stats)

            # 控制播放速度
            time.sleep(max(0, 1/30 - elapsed))

        self.is_running = False
        self.root.after(0, lambda: self.play_btn.config(text="▶  开始"))

    def _annotate(self, frame, result):
        """在帧上绘制检测框，返回标注后的帧和统计字典"""
        stats = defaultdict(int)
        names = self.model.names

        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = names[cls_id]
            stats[name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS_BGR.get(name, (200, 200, 200))

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            label = f"{name}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)

            # 标签文字
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame, stats

    # ══════════════════════════════════════════════════════════
    #  画面显示
    # ══════════════════════════════════════════════════════════
    def _show_frame(self, frame):
        """将 OpenCV 帧显示到 tkinter Canvas 上"""
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            h, w = frame.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((nw, nh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            x, y = (cw - nw) // 2, (ch - nh) // 2
            self.canvas.create_image(x, y, anchor='nw', image=photo)
            self.canvas._photo = photo  # 防止被垃圾回收
        except Exception:
            pass

    def _on_canvas_resize(self, event):
        pass  # 实时缩放由 _show_frame 处理

    # ══════════════════════════════════════════════════════════
    #  统计面板刷新
    # ══════════════════════════════════════════════════════════
    def _refresh_stats(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()

        all_classes = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']
        icons = {'person': '🚶', 'car': '🚗', 'bus': '🚌',
                 'truck': '🚛', 'bicycle': '🚲', 'motorcycle': '🏍️'}

        for cls in all_classes:
            count = self.result_stats.get(cls, 0)
            color = COLORS['success'] if count > 0 else COLORS['subtext']

            row = tk.Frame(self.stats_frame, bg=COLORS['card'])
            row.pack(fill='x', pady=2)

            tk.Label(row, text=f"{icons.get(cls, '●')} {cls}",
                     font=('Helvetica', 9), bg=COLORS['card'],
                     fg=color, anchor='w', width=14).pack(side='left')
            tk.Label(row, text=str(count),
                     font=('Helvetica', 9, 'bold'), bg=COLORS['card'],
                     fg=color, anchor='e').pack(side='right')

        total = sum(self.result_stats.values())
        sep = tk.Frame(self.stats_frame, bg=COLORS['border'], height=1)
        sep.pack(fill='x', pady=4)
        total_row = tk.Frame(self.stats_frame, bg=COLORS['card'])
        total_row.pack(fill='x')
        tk.Label(total_row, text="总计",
                 font=('Helvetica', 9, 'bold'), bg=COLORS['card'],
                 fg=COLORS['text'], anchor='w').pack(side='left')
        tk.Label(total_row, text=str(total),
                 font=('Helvetica', 11, 'bold'), bg=COLORS['card'],
                 fg=COLORS['accent'], anchor='e').pack(side='right')

    # ══════════════════════════════════════════════════════════
    #  保存结果
    # ══════════════════════════════════════════════════════════
    def _save_result(self):
        if not hasattr(self, '_last_annotated') or self._last_annotated is None:
            self.status_var.set("⚠️ 还没有检测结果可以保存")
            return

        path = filedialog.asksaveasfilename(
            title="保存结果图片",
            defaultextension=".jpg",
            filetypes=[("JPEG图片", "*.jpg"), ("PNG图片", "*.png")]
        )
        if path:
            cv2.imwrite(path, self._last_annotated)

            # 同时保存统计 JSON
            json_path = os.path.splitext(path)[0] + '_stats.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.result_stats), f, ensure_ascii=False, indent=2)

            self.status_var.set(f"✅ 结果已保存：{path}  统计数据：{json_path}")


# ══════════════════════════════════════════════════════════════
#  程序入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()