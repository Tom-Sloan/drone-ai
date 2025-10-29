"""
2D Joystick Visualization Widget
Displays RC channel positions as a visual joystick with X/Y axes
"""

import tkinter as tk
from tkinter import ttk


class JoystickWidget(tk.Canvas):
    """
    A 2D joystick visualization widget
    Displays position on X and Y axes from -1 to 1
    """

    def __init__(self, parent, size=150, title="Joystick", **kwargs):
        super().__init__(parent, width=size, height=size, bg='#2b2b2b', highlightthickness=1,
                         highlightbackground='#555555', **kwargs)

        self.size = size
        self.title = title
        self.center = size / 2
        self.radius = (size - 20) / 2  # Joystick bounds radius

        # Current position (-1 to 1)
        self.x_pos = 0.0
        self.y_pos = 0.0

        # Draw the joystick base
        self._draw_base()

        # Create the position indicator (will be moved)
        self.indicator = None
        self._draw_indicator()

    def _draw_base(self):
        """Draw the joystick base with axes and grid"""
        # Draw outer circle (bounds)
        padding = 10
        self.create_oval(
            padding, padding,
            self.size - padding, self.size - padding,
            outline='#555555', width=2
        )

        # Draw center crosshairs
        self.create_line(
            self.center, padding,
            self.center, self.size - padding,
            fill='#444444', width=1, dash=(2, 4)
        )
        self.create_line(
            padding, self.center,
            self.size - padding, self.center,
            fill='#444444', width=1, dash=(2, 4)
        )

        # Draw center dot
        center_size = 3
        self.create_oval(
            self.center - center_size, self.center - center_size,
            self.center + center_size, self.center + center_size,
            fill='#666666', outline='#666666'
        )

        # Add axis labels
        self.create_text(
            self.size / 2, 5,
            text='+Y', fill='#888888', font=('Arial', 8)
        )
        self.create_text(
            self.size / 2, self.size - 5,
            text='-Y', fill='#888888', font=('Arial', 8)
        )
        self.create_text(
            5, self.size / 2,
            text='-X', fill='#888888', font=('Arial', 8)
        )
        self.create_text(
            self.size - 5, self.size / 2,
            text='+X', fill='#888888', font=('Arial', 8)
        )

    def _draw_indicator(self):
        """Draw the position indicator"""
        indicator_size = 8
        x_pixel = self.center + (self.x_pos * self.radius)
        y_pixel = self.center - (self.y_pos * self.radius)  # Invert Y for screen coords

        # Delete old indicator if exists
        if self.indicator:
            self.delete(self.indicator)
            if hasattr(self, 'indicator_outline'):
                self.delete(self.indicator_outline)

        # Draw outline
        self.indicator_outline = self.create_oval(
            x_pixel - indicator_size - 1, y_pixel - indicator_size - 1,
            x_pixel + indicator_size + 1, y_pixel + indicator_size + 1,
            outline='white', width=1
        )

        # Draw indicator
        self.indicator = self.create_oval(
            x_pixel - indicator_size, y_pixel - indicator_size,
            x_pixel + indicator_size, y_pixel + indicator_size,
            fill='#00ff00', outline='#00cc00', width=2
        )

    def update_position(self, x, y):
        """
        Update joystick position
        x, y should be in range -1.0 to 1.0
        """
        # Clamp values to -1 to 1
        self.x_pos = max(-1.0, min(1.0, x))
        self.y_pos = max(-1.0, min(1.0, y))

        # Redraw indicator
        self._draw_indicator()

    def update_from_rc(self, x_channel, y_channel):
        """
        Update position from RC channel values (1000-2000 range)
        1500 = center (0.0)
        1000 = -1.0
        2000 = +1.0
        """
        # Convert RC values (1000-2000) to normalized (-1 to 1)
        x_normalized = (x_channel - 1500) / 500.0
        y_normalized = (y_channel - 1500) / 500.0

        self.update_position(x_normalized, y_normalized)

    def get_position_text(self):
        """Get position as formatted text"""
        return f"X: {self.x_pos:+.2f}  Y: {self.y_pos:+.2f}"
