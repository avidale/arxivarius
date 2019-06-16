

class Response:
    def __init__(self, text, buttons=None):
        self.text = text
        self.buttons = buttons or []
