import pygame

class PyGameDisplay():
    def __init__(
        self,
        width=800,
        height=600
    ) -> None:
        pygame.init()

        self.display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.font = self._get_font()
        self.clock = pygame.time.Clock()
    
    def _get_font(self):
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = "ubuntumono"
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False
            