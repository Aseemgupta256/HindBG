# bg_remover.py - Enhanced Background Removal with Trimap-based Alpha Refinement
import os
import logging
import torch
import numpy as np
import cv2
import gc
import io
import mimetypes
import tempfile
import shutil
import uuid
from dotenv import load_dotenv
load_dotenv()
from PIL import Image, ImageOps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_halo_artifacts(alpha, base_kernel_size=3, blend_weight=0.3):
    """
    Remove halo artifacts using dynamic kernel sizing, bilateral filtering,
    erosion, and an additional median blur.
    
    Args:
        alpha: Alpha channel as a uint8 image (values 0-255).
        base_kernel_size: Minimum kernel size.
        blend_weight: Weight factor for blending the eroded and original alpha.
    
    Returns:
        Decontaminated alpha channel as a uint8 image.
    """
    # Dynamically determine kernel size from image dimensions
    dynamic_kernel = max(base_kernel_size, int(min(alpha.shape[:2]) / 200))
    
    # Use bilateral filtering to smooth while preserving edges
    alpha_bilateral = cv2.bilateralFilter(alpha, d=dynamic_kernel*2+1, sigmaColor=50, sigmaSpace=50)
    
    # Create an elliptical kernel for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic_kernel, dynamic_kernel))
    eroded = cv2.erode(alpha_bilateral, kernel, iterations=1)
    
    # Blend the original alpha with the eroded alpha
    blended = cv2.addWeighted(alpha, 1 - blend_weight, eroded, blend_weight, 0)
    
    # Apply additional median filtering to remove small artifacts
    final_alpha = cv2.medianBlur(blended, 3)
    final_alpha = np.clip(final_alpha, 0, 255).astype(np.uint8)
    return final_alpha

def generate_trimap(alpha, low_thresh=15, high_thresh=240):
    """
    Generate a trimap from the given alpha mask.
    
    Args:
        alpha: Alpha mask (uint8 image).
        low_thresh: Threshold below which is considered definite background.
        high_thresh: Threshold above which is considered definite foreground.
    
    Returns:
        Trimap image (0 for background, 255 for foreground, 128 for uncertain).
    """
    trimap = np.zeros_like(alpha)
    trimap[alpha >= high_thresh] = 255
    trimap[alpha <= low_thresh] = 0
    uncertain = (alpha > low_thresh) & (alpha < high_thresh)
    trimap[uncertain] = 128
    return trimap

def refine_alpha_channel_advanced(alpha_np, guidance_img, radius=4, eps=1e-2):
    """
    Advanced refinement of the alpha channel.
    
    This function generates a trimap from the initial alpha,
    then applies a guided filter only on the uncertain areas,
    and finally blends the refined uncertain regions back with the original.
    
    Args:
        alpha_np: Initial alpha channel as a uint8 numpy array.
        guidance_img: Corresponding RGB image as a uint8 numpy array.
        radius: Radius for the guided filter.
        eps: Regularization parameter for the guided filter.
    
    Returns:
        The refined alpha channel as a uint8 numpy array.
    """
    # Generate a trimap: definite foreground (255), background (0), uncertain (128)
    trimap = generate_trimap(alpha_np)
    
    # Convert alpha to float [0, 1]
    alpha_float = alpha_np.astype(np.float32) / 255.0
    
    # Convert guidance image to grayscale and normalize to [0, 1]
    guide_gray = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Initialize refined_alpha with the original
    refined_alpha = alpha_float.copy()
    
    # Identify uncertain regions (trimap == 128)
    uncertain_mask = (trimap == 128)
    if np.any(uncertain_mask):
        try:
            # Apply guided filter to the entire alpha (as uint8 scaled to 0-255)
            guided = cv2.ximgproc.guidedFilter(guide_gray, (alpha_float * 255).astype(np.uint8), radius, eps)
            guided = guided.astype(np.float32) / 255.0
        except AttributeError:
            # If guided filter is not available, fallback to morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            guided = cv2.morphologyEx((alpha_float * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            guided = guided.astype(np.float32) / 255.0
        
        # Replace uncertain areas in refined_alpha with the guided result
        refined_alpha[uncertain_mask] = guided[uncertain_mask]
    
    # Scale back to [0, 255] and convert to uint8
    refined_alpha_uint8 = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
    
    # Optionally, apply an additional median blur to reduce noise
    refined_alpha_uint8 = cv2.medianBlur(refined_alpha_uint8, 3)
    
    # Finally, run the halo removal function on the refined alpha
    alpha_final = remove_halo_artifacts(refined_alpha_uint8, base_kernel_size=3, blend_weight=0.3)
    return alpha_final

class BackgroundRemover:
    """
    Background remover using the InSPyReNet model from the transparent-background package.
    Integrates dynamic padding and the new trimap-based alpha refinement for improved accuracy.
    """
    def __init__(self, use_jit=None):
        try:
            if use_jit is None:
                use_jit = os.getenv('USE_JIT', 'true').lower() == 'true'
            from transparent_background import Remover
            if torch.cuda.is_available() and os.getenv("USE_GPU", "false").lower() == 'true':
                device = "cuda"
                logger.info("Using GPU acceleration with CUDA")
            else:
                device = "cpu"
                logger.info("Using CPU for processing")
            self.remover = Remover(device=device, jit=use_jit)
            self.temp_files = []
            logger.info("Background remover initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing remover: {str(e)}")
            raise

    def __del__(self):
        self.cleanup_temp_files()

    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                        logger.debug(f"Cleaned up temporary directory: {temp_file}")
                    else:
                        os.remove(temp_file)
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_file}: {str(e)}")
        self.temp_files = []

    def remove_background(self, image_path, output_dir):
        """
        Remove the background from an image with advanced edge refinement.
        Uses dynamic padding (5% of the smallest dimension) to preserve details.
    
        Args:
            image_path: Path to the input image.
            output_dir: Directory to save the processed image.
    
        Returns:
            Path to the saved processed PNG image.
        """
        try:
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if start_time:
                start_time.record()
            logger.info(f"Processing image: {image_path}")
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'image/jpeg'
            logger.info(f"Image mime type: {mime_type}")
            if 'heic' in image_path.lower() or 'heif' in image_path.lower():
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                    logger.info("HEIF/HEIC support enabled")
                except ImportError:
                    logger.warning("HEIF/HEIC support not available. Install pillow_heif.")
            temp_dir = tempfile.mkdtemp(prefix="bg_remover_")
            self.temp_files.append(temp_dir)
            try:
                img = Image.open(image_path)
            except Exception as e:
                logger.error(f"Error opening image with PIL: {str(e)}")
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    img = Image.open(io.BytesIO(img_data))
            img = ImageOps.exif_transpose(img)
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
                logger.info("Converted image to RGB")
            original_size = img.size
            min_dimension = 256   # Scale up only if image is too small
            resized = False
            if min(original_size) < min_dimension:
                scale_factor = min_dimension / min(original_size)
                new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                img = img.resize(new_size, Image.LANCZOS)
                resized = True
                logger.info(f"Scaled up small image from {original_size} to {img.size}")

            # --- Dynamic Padding: Use 5% of the smallest dimension
            padding = int(min(img.size) * 0.05)
            img_np = np.array(img)
            img_padded_np = cv2.copyMakeBorder(img_np, padding, padding, padding, padding, borderType=cv2.BORDER_REPLICATE)
            img_padded = Image.fromarray(img_padded_np)
            logger.info(f"Added dynamic padding of {padding}px to avoid cropping details.")

            logger.info(f"Processing at dimensions (with padding): {img_padded.size}")
            logger.info("Removing background")
            processed_array = self.remover.process(img_padded, type='rgba')
            if not isinstance(processed_array, np.ndarray):
                logger.warning("Expected numpy array, converting...")
                processed_array = np.array(processed_array)
            if processed_array.dtype == np.float32:
                processed_array = (processed_array * 255).astype(np.uint8)
            if len(processed_array.shape) == 3 and processed_array.shape[2] == 4:
                rgb = processed_array[:, :, :3]
                alpha_np = processed_array[:, :, 3]
                guidance = np.array(img_padded)
                alpha_refined = refine_alpha_channel_advanced(alpha_np, guidance, radius=4, eps=1e-2)
                processed_array = np.dstack((rgb, alpha_refined))
                processed_img = Image.fromarray(processed_array)
            else:
                logger.warning("Unexpected output shape from remover")
                processed_img = Image.fromarray(processed_array)

            # --- Crop out the dynamic padding
            width, height = processed_img.size
            processed_img = processed_img.crop((padding, padding, width - padding, height - padding))
            logger.info("Removed dynamic padding after processing.")

            if resized:
                processed_img = processed_img.resize(original_size, Image.LANCZOS)
                logger.info(f"Restored original dimensions: {original_size}")
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{uuid.uuid4()}.png"
            output_path = os.path.join(output_dir, filename)
            processed_img.save(output_path, format="PNG", optimize=True)
            logger.info(f"Saved processed image to: {output_path}")
            del processed_array, processed_img
            gc.collect()
            self.cleanup_temp_files()
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000
                logger.info(f"Processing completed in {elapsed_time:.2f} seconds (GPU)")
            return output_path

        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.cleanup_temp_files()
            raise Exception(f"Error processing image: {str(e)}")

    def batch_process(self, image_paths, output_dir):
        """
        Process multiple images sequentially.
    
        Args:
            image_paths: List of paths to input images.
            output_dir: Directory to save output images.
    
        Returns:
            List of paths to processed images.
        """
        output_paths = []
        for path in image_paths:
            try:
                output_path = self.remove_background(path, output_dir)
                output_paths.append(output_path)
                logger.info(f"Successfully processed {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")
                continue
        return output_paths

