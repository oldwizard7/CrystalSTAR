"""Environment variable loader for API keys"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_env_file(env_path: str = None):
    """
    Load environment variables from an env file.

    Args:
        env_path: Optional explicit path to env file.
    """
    try:
        from dotenv import load_dotenv

        if env_path is None:
            env_override = os.getenv("MATAGENT_ENV_FILE")
            if env_override:
                env_path = Path(env_override).expanduser()
            else:
                home_env = Path.home() / ".matagent_env"
                if home_env.exists():
                    env_path = home_env
                else:
                    # Get project root (3 levels up from this file)
                    project_root = Path(__file__).resolve().parents[2]
                    env_path = project_root / ".env"

        # Load .env file and override existing environment variables.
        # This prevents stale shell variables from shadowing MATAGENT_ENV_FILE.
        if Path(env_path).exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment from: {env_path}")
            return True
        else:
            logger.warning(f".env file not found at: {env_path}")
            return False

    except ImportError:
        logger.error("python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    except Exception as e:
        logger.error(f"Failed to load .env file: {e}")
        return False


def get_api_key(key_name: str, required: bool = True) -> str:
    """
    Get API key from environment

    Args:
        key_name: Name of the environment variable
        required: If True, raise error if not found

    Returns:
        API key value

    Raises:
        ValueError: If required=True and key not found
    """
    value = os.getenv(key_name)

    if value is None or value == "" or value.startswith("your_"):
        if required:
            raise ValueError(
                f"{key_name} not set!\n"
                f"Please set it in .env file or environment:\n"
                f"  echo '{key_name}=your_actual_key' >> .env"
            )
        else:
            logger.warning(f"{key_name} not set (optional)")
            return None

    return value


def check_environment():
    """
    Check if all required environment variables are set

    Returns:
        True if all required keys are present
    """
    logger.info("Checking environment variables...")

    # Load .env if exists
    load_env_file()

    all_ok = True

    # Check OpenAI API key (required)
    try:
        openai_key = get_api_key('OPENAI_API_KEY', required=True)
        logger.info(f"✓ OPENAI_API_KEY: {openai_key[:10]}...")
    except ValueError as e:
        logger.error(f"✗ OPENAI_API_KEY: {e}")
        all_ok = False

    # Check MP API key (optional)
    try:
        mp_key = get_api_key('MP_API_KEY', required=False)
        if mp_key:
            logger.info(f"✓ MP_API_KEY: {mp_key[:10]}...")
        else:
            logger.info("○ MP_API_KEY: Not set (optional)")
    except ValueError:
        pass

    return all_ok


# Auto-load when module is imported
load_env_file()


if __name__ == "__main__":
    # Test the environment loader
    import sys

    print("="*60)
    print("Environment Variable Check")
    print("="*60)

    if check_environment():
        print("\n✅ All required environment variables are set!")
        sys.exit(0)
    else:
        print("\n❌ Some required environment variables are missing!")
        print("\nPlease edit .env file and add your API keys:")
        print("  nano .env")
        sys.exit(1)
