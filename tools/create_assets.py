from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/SFNS.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_badge(filename: str, title: str, emoji: str, background: tuple[int, int, int]) -> None:
    size = 640
    image = Image.new("RGBA", (size, size), (*background, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((26, 26, size - 26, size - 26), radius=100, fill=(*background, 235), outline=(255, 255, 255, 150), width=10)
    draw.ellipse((130, 110, 510, 490), fill=(255, 255, 255, 42), outline=(255, 255, 255, 185), width=8)
    emoji_font = font(190)
    title_font = font(62)
    emoji_box = draw.textbbox((0, 0), emoji, font=emoji_font)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((size - (emoji_box[2] - emoji_box[0])) / 2, 155), emoji, font=emoji_font, fill=(255, 255, 255, 255))
    draw.text(((size - (title_box[2] - title_box[0])) / 2, 510), title, font=title_font, fill=(255, 255, 255, 255))
    image.save(ASSETS / filename)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    make_badge("positive.png", "POSITIVE", "+", (35, 170, 95))
    make_badge("negative.png", "NEGATIVE", "!", (205, 65, 78))
