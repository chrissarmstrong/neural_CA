# CA: Courtesy of Deepseek Coder v2 (with a few tweaks)

import tkinter as tk
import numpy as np

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing App")
        self.canvas_size = 28
        self.cell_size = 10
        self.canvas = tk.Canvas(master, width=self.canvas_size * self.cell_size, height=self.canvas_size * self.cell_size)
        self.canvas.pack()

        self.color = (255, 255, 255, 0)  # Default color is white with 0 alpha
        self.drawing = np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.uint8)
        self.drawing[:, :, :3] = 255  # Set RGB channels to white
        
        self.draw_grid()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)

        self.color_frame = tk.Frame(master)
        self.color_frame.pack()
        self.red_scale = tk.Scale(self.color_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Red")
        self.red_scale.pack(side=tk.LEFT)
        self.green_scale = tk.Scale(self.color_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Green")
        self.green_scale.set(255)
        self.green_scale.pack(side=tk.LEFT)
        self.blue_scale = tk.Scale(self.color_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Blue")
        self.blue_scale.pack(side=tk.LEFT)

        self.button = tk.Button(master, text="Get Drawing", command=self.get_drawing)
        self.button.pack()

    def draw_grid(self):
        for i in range(self.canvas_size):
            for j in range(self.canvas_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")

    def start_draw(self, event):
        self.draw(event)

    def draw(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            r, g, b = self.red_scale.get(), self.green_scale.get(), self.blue_scale.get()
            self.color = (r, g, b, 255)  # Set alpha to 255 for drawn pixels
            self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size, (x + 1) * self.cell_size, (y + 1) * self.cell_size, fill=self.rgb_to_hex(self.color[:3]), outline="gray")
            self.drawing[y, x] = self.color

    def rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def get_drawing(self):
        self.master.quit()

def get_drawing_from_ui():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
    return app.drawing
