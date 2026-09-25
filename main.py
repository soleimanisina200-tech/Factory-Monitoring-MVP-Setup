import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Industrial CV Factory Monitoring MVP...")
    # Future phases will initialize streams, detection, and tracking here
    logger.info("System shut down securely.")

if __name__ == "__main__":
    main()