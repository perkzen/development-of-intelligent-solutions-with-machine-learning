from pydantic import BaseModel


class ImageData(BaseModel):
    image_base64: str
