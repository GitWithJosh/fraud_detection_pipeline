from fraud_detection_service import FraudDetectionService

import tkinter as tk
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

if __name__ == "__main__":    
    fds = FraudDetectionService("model.onnx")    
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
