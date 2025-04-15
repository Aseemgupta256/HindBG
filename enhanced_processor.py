# enhanced_processor.py - Enhanced Image Processor with Adjustments for Clearer Edge Details

import os
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedProcessor:
    """
    Enhanced image processor using PIL that sharpens details while taking care not to over-enhance edges.
    """
    
    def __init__(self, app_dir):
        self.app_dir = app_dir
        logger.info("Enhanced processor initialized")
    
    def process_image(self, input_path, output_dir, reality_ratio=0.5):
        """
        Process image with enhancements and background removal.
    
        Args:
            input_path: Path to the input image.
            output_dir: Directory to save the output image.
            reality_ratio: Fraction of the original to blend with enhancements (0-1).
        """
        try:
            # Enhance image first
            enhanced_path = self._enhance_image(input_path, output_dir, reality_ratio)
            
            # Then remove background
            from bg_remover import BackgroundRemover
            logger.info("Initializing background remover for enhanced image")
            remover = BackgroundRemover()
            final_output = remover.remove_background(enhanced_path, output_dir)
            
            # Clean up intermediate file if applicable
            if os.path.exists(enhanced_path) and enhanced_path != input_path:
                try:
                    os.remove(enhanced_path)
                except Exception as e:
                    logger.warning(f"Failed to remove intermediate file: {e}")
            
            return final_output

        except Exception as e:
            logger.error(f"Error in enhancement process: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to direct background removal
            try:
                logger.info("Falling back to direct background removal")
                from bg_remover import BackgroundRemover
                remover = BackgroundRemover()
                return remover.remove_background(input_path, output_dir)
            except Exception as bg_error:
                logger.error(f"Background removal also failed: {str(bg_error)}")
                logger.error(traceback.format_exc())
                raise Exception(f"Image processing failed: {str(e)}")
    
    def _enhance_image(self, input_path, output_dir, reality_ratio):
        """
        Apply enhancements focusing on detail while preserving edge information.
    
        Args:
            input_path: Path to the input image.
            output_dir: Directory for saving the enhanced output.
            reality_ratio: Control ratio to blend original with enhanced version.
            
        Returns:
            Path to the enhanced image.
        """
        try:
            logger.info(f"Enhancing image with reality ratio: {reality_ratio}")
            img = Image.open(input_path).convert('RGB')
            original = img.copy()

            if min(img.width, img.height) < 512:
                scale = max(1.5, 512 / min(img.width, img.height))
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Upscaled small image to {new_width}x{new_height}")

            img = img.filter(ImageFilter.UnsharpMask(radius=5, percent=150, threshold=3))
            logger.info("Applied unsharp mask with radius 5")

            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.15)
            logger.info("Applied 15% contrast boost")

            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.05)
            logger.info("Applied 5% color boost")

            # Decrease shadows by 10%
            img_hsv = img.convert('HSV')
            h, s, v = img_hsv.split()
            v_array = np.array(v).astype(np.float32)
            shadow_mask = v_array < 128
            v_array[shadow_mask] = np.clip(v_array[shadow_mask] * 1.1, 0, 255)
            v_new = Image.fromarray(v_array.astype(np.uint8))
            img_hsv = Image.merge('HSV', (h, s, v_new))
            img = img_hsv.convert('RGB')
            logger.info("Decreased shadows by 10%")

            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.03)
            logger.info("Applied 3% brightness boost")

            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.6)
            logger.info("Applied 60% sharpness boost")

            if img.size != original.size:
                img = img.resize(original.size, Image.LANCZOS)

            actual_reality_ratio = max(0.6, reality_ratio)
            if actual_reality_ratio < 1.0:
                original_array = np.array(original).astype(np.float32)
                enhanced_array = np.array(img).astype(np.float32)
                blended_array = (original_array * actual_reality_ratio + 
                                 enhanced_array * (1 - actual_reality_ratio)).astype(np.uint8)
                result = Image.fromarray(blended_array)
                logger.info(f"Blended with {actual_reality_ratio*100:.0f}% original and {(1-actual_reality_ratio)*100:.0f}% enhancements")
            else:
                result = original
                logger.info("Using 100% original image (no enhancement applied)")

            enhanced_path = os.path.join(output_dir, f"enhanced_{os.path.basename(input_path)}")
            result.save(enhanced_path, quality=95)
            return enhanced_path
        except Exception as e:
            logger.error(f"Enhancement error: {str(e)}")
            return input_path

    def enhance_image(self, input_path, output_dir, reality_ratio=0.5):
        """
        Enhance the image without performing background removal.
    
        Args:
            input_path: Path to the input image.
            output_dir: Directory to save enhanced image.
            reality_ratio: Enhancement ratio control.
            
        Returns:
            Path to the enhanced image.
        """
        try:
            return self._enhance_image(input_path, output_dir, reality_ratio)
        except Exception as e:
            logger.error(f"Error in enhancement process: {str(e)}")
            raise
