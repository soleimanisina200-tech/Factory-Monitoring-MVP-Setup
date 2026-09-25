import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostics():
    logger.info("--- Running System Diagnostics ---")
    
    # 1. Check Python Version
    logger.info(f"Python Version: {sys.version.split(' ')[0]}")
    if sys.version_info < (3, 11):
        logger.warning("Python version is below 3.11. Please upgrade.")

    # 2. Check PyTorch and CUDA
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA Device Count: {device_count}")
            logger.info(f"Current CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            logger.error("CUDA is NOT available. AI models will run on CPU, which is too slow for real-time monitoring.")
    except ImportError:
        logger.error("PyTorch is not installed.")

    # 3. Check OpenCV
    try:
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV is not installed.")

    # 4. Check Ultralytics (YOLO)
    try:
        import ultralytics
        logger.info(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        logger.error("Ultralytics is not installed.")

    logger.info("--- Diagnostics Complete ---")

if __name__ == "__main__":
    run_diagnostics()