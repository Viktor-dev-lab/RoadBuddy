import cv2
from pathlib import Path
from typing import List

class ImageLoader:
    """Image loading utilities"""
    
    @staticmethod
    def load_image(path: Path):
        """Load image từ path"""
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return image
    
    @staticmethod
    def save_image(image, path: Path):
        """Save image"""
        cv2.imwrite(str(path), image)
    
    @staticmethod
    def get_image_files(directory: Path) -> List[Path]:
        """Lấy tất cả image files trong thư mục"""
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        files = []
        for ext in extensions:
            files.extend(directory.glob(ext))
        return sorted(files)