"""
Camera Widget for displaying USB camera feed in tkinter
Supports USB2.0 PC CAMERA ultrathink and other compatible cameras
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time


class CameraWidget(ttk.Frame):
    """Widget for displaying live camera feed from USB camera"""

    def __init__(self, parent, width=400, height=300, camera_index=0, fps=30):
        """
        Initialize camera widget

        Args:
            parent: Parent tkinter widget
            width: Display width in pixels
            height: Display height in pixels
            camera_index: Camera device index (0 for first camera, 1 for second, etc.)
            fps: Target frames per second for display
        """
        super().__init__(parent)

        self.width = width
        self.height = height
        self.camera_index = camera_index
        self.fps = fps
        self.frame_delay = int(1000 / fps)  # milliseconds between frames

        # Camera state
        self.camera = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Camera info
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

        # Build UI
        self._build_ui()

        # Start camera
        self._start_camera()

    def _build_ui(self):
        """Build the camera display UI"""
        # Camera display label
        self.canvas = tk.Label(self, bg='black', width=self.width, height=self.height)
        self.canvas.pack()

        # Status label
        self.status_label = ttk.Label(
            self,
            text="Initializing camera...",
            font=('Arial', 9),
            foreground='gray'
        )
        self.status_label.pack(pady=2)

        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill='x', pady=2)

        # Start/Stop button
        self.toggle_btn = ttk.Button(
            control_frame,
            text="Stop Camera",
            command=self._toggle_camera,
            width=15
        )
        self.toggle_btn.pack(side='left', padx=5)

        # Camera selector
        ttk.Label(control_frame, text="Camera:").pack(side='left', padx=5)
        self.camera_var = tk.StringVar(value=str(self.camera_index))
        self.camera_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            width=3,
            state='readonly',
            values=['0', '1', '2', '3']
        )
        self.camera_dropdown.pack(side='left')
        self.camera_dropdown.bind('<<ComboboxSelected>>', self._change_camera)

    def _start_camera(self):
        """Start the camera capture"""
        if self.running:
            return

        self.running = True

        # Start camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

        # Start display update loop
        self._update_display()

    def _stop_camera(self):
        """Stop the camera capture"""
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # Clear display
        self.canvas.config(image='', bg='black')
        self.status_label.config(text="Camera stopped", foreground='gray')

    def _camera_loop(self):
        """Background thread for camera capture"""
        try:
            # Open camera
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                self._update_status("Camera not found", 'red')
                self.running = False
                return

            # Set camera properties (request specific resolution)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            # Get actual camera properties (what the camera actually provides)
            self.actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend_name = self.camera.getBackendName()

            # Display initial status with resolution
            self._update_status(
                f"Camera {self.camera_index} ({self.actual_width}x{self.actual_height}) - {backend_name}",
                'green'
            )

            frame_count = 0
            start_time = time.time()

            while self.running:
                ret, frame = self.camera.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Store frame
                    with self.frame_lock:
                        self.current_frame = frame_rgb

                    # Calculate FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        self.actual_fps = 30 / elapsed
                        self._update_status(
                            f"Camera {self.camera_index}: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f} FPS",
                            'green'
                        )
                        start_time = time.time()
                else:
                    self._update_status("Camera read error", 'red')
                    time.sleep(0.1)

                # Small delay to control capture rate
                time.sleep(0.01)

        except Exception as e:
            self._update_status(f"Camera error: {str(e)}", 'red')

        finally:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    def _update_display(self):
        """Update the display with the latest camera frame"""
        if not self.running:
            return

        with self.frame_lock:
            if self.current_frame is not None:
                # Resize frame to fit display
                frame_resized = cv2.resize(
                    self.current_frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR
                )

                # Convert to PIL Image
                img = Image.fromarray(frame_resized)

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=img)

                # Update label
                self.canvas.config(image=photo, bg='black')
                self.canvas.image = photo  # Keep a reference

        # Schedule next update
        self.after(self.frame_delay, self._update_display)

    def _update_status(self, text, color):
        """Update status label (thread-safe)"""
        def update():
            self.status_label.config(text=text, foreground=color)

        self.after(0, update)

    def _toggle_camera(self):
        """Toggle camera on/off"""
        if self.running:
            self._stop_camera()
            self.toggle_btn.config(text="Start Camera")
        else:
            self._start_camera()
            self.toggle_btn.config(text="Stop Camera")

    def _change_camera(self, event=None):
        """Change camera device"""
        new_index = int(self.camera_var.get())

        if new_index != self.camera_index:
            # Stop current camera
            self._stop_camera()

            # Update index
            self.camera_index = new_index

            # Restart camera
            time.sleep(0.2)
            self._start_camera()
            self.toggle_btn.config(text="Stop Camera")

    def cleanup(self):
        """Cleanup resources when closing"""
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
