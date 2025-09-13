"""Configuration management for the Deep Podcast system.

This module provides centralized configuration management with support for:
- Environment variable loading
- Configuration validation
- Default value handling
- Runtime configuration updates
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from shared.models import LLMConfig, TTSConfig, SearchConfig, OutputConfig, SystemConfig


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration settings."""
    
    # Environment detection
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    
    # Security
    allowed_hosts: list[str] = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost").split(","))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key"))
    
    # Performance
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_TASKS", "5")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))
    
    # Storage
    temp_dir: str = field(default_factory=lambda: os.getenv("TEMP_DIR", "/tmp/deep_podcast"))
    cleanup_interval: int = field(default_factory=lambda: int(os.getenv("CLEANUP_INTERVAL", "3600")))  # 1 hour


class ConfigurationManager:
    """Central configuration manager for the Deep Podcast system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self._system_config: Optional[SystemConfig] = None
        self._env_config: Optional[EnvironmentConfig] = None
        self._loaded = False
        
    def load_configuration(self) -> SystemConfig:
        """Load and validate complete system configuration.
        
        Returns:
            Complete system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            if self._loaded and self._system_config:
                return self._system_config
            
            logger.info("Loading system configuration...")
            
            # Load environment variables
            self._load_environment_variables()
            
            # Create component configurations
            llm_config = self._create_llm_config()
            tts_config = self._create_tts_config()
            search_config = self._create_search_config()
            output_config = self._create_output_config()
            
            # Create system configuration
            self._system_config = SystemConfig(
                llm_config=llm_config,
                tts_config=tts_config,
                search_config=search_config,
                output_config=output_config
            )
            
            # Validate configuration
            self._validate_configuration(self._system_config)
            
            # Setup directories
            self._setup_directories(self._system_config)
            
            self._loaded = True
            logger.info("System configuration loaded successfully")
            
            return self._system_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment-specific configuration."""
        if not self._env_config:
            self._env_config = EnvironmentConfig()
        return self._env_config
    
    def _load_environment_variables(self):
        """Load environment variables from .env file if present."""
        env_file = Path(".env")
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file loading")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
    
    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "anthropic")
        
        # Model selection based on provider
        if provider == "anthropic":
            default_model = "claude-3-5-sonnet-20240620"
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif provider == "openai":
            default_model = "gpt-4-turbo-preview"
            # For OpenAI provider, try OPENAI_API_KEY first, then fallback to LLM_API_KEY (for custom APIs like DashScope)
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
        else:
            default_model = "claude-3-5-sonnet-20240620"
            api_key = os.getenv("LLM_API_KEY", "")
        
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model),
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60"))
        )
    
    def _create_tts_config(self) -> TTSConfig:
        """Create TTS configuration from environment variables."""
        voice_models_str = os.getenv("TTS_VOICE_MODELS", "")
        
        if voice_models_str:
            try:
                import json
                voice_models = json.loads(voice_models_str)
            except json.JSONDecodeError:
                voice_models = {"host": "xiaoyun", "guest": "xiaogang"}
        else:
            voice_models = {"host": "xiaoyun", "guest": "xiaogang"}
        
        return TTSConfig(
            provider=os.getenv("TTS_PROVIDER", "qwen"),
            api_key=os.getenv("QWEN_TTS_API_KEY", ""),
            base_url=os.getenv("QWEN_TTS_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2speech/synthesis"),
            voice_models=voice_models,
            audio_format=os.getenv("TTS_AUDIO_FORMAT", "mp3"),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "16000")),
            speech_rate=float(os.getenv("TTS_SPEECH_RATE", "1.0")),
            pitch=int(os.getenv("TTS_PITCH", "0")),
            volume=float(os.getenv("TTS_VOLUME", "1.0"))
        )
    
    def _create_search_config(self) -> SearchConfig:
        """Create search configuration from environment variables."""
        return SearchConfig(
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "20")),
            max_crawl_pages=int(os.getenv("MAX_CRAWL_PAGES", "50")),
            search_timeout=int(os.getenv("SEARCH_TIMEOUT", "30")),
            crawl_timeout=int(os.getenv("CRAWL_TIMEOUT", "15")),
            content_min_length=int(os.getenv("CONTENT_MIN_LENGTH", "500")),
            concurrent_requests=int(os.getenv("CONCURRENT_REQUESTS", "5"))
        )
    
    def _create_output_config(self) -> OutputConfig:
        """Create output configuration from environment variables."""
        return OutputConfig(
            base_dir=os.getenv("OUTPUT_BASE_DIR", "./outputs"),
            report_format=os.getenv("REPORT_FORMAT", "markdown"),
            audio_format=os.getenv("AUDIO_FORMAT", "mp3"),
            audio_quality=os.getenv("AUDIO_QUALITY", "high"),
            file_naming=os.getenv("FILE_NAMING", "{timestamp}_{topic_hash}"),
            compression=os.getenv("COMPRESSION", "true").lower() == "true",
            cleanup_temp=os.getenv("CLEANUP_TEMP", "true").lower() == "true"
        )
    
    def _validate_configuration(self, config: SystemConfig):
        """Validate the loaded configuration.
        
        Args:
            config: System configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []
        
        # Validate LLM configuration
        if not config.llm_config.api_key:
            errors.append("LLM API key is required")
        
        if config.llm_config.temperature < 0 or config.llm_config.temperature > 2:
            errors.append("LLM temperature must be between 0 and 2")
        
        # Validate TTS configuration
        if config.tts_config.provider == "qwen" and not config.tts_config.api_key:
            errors.append("Qwen TTS API key is required when using qwen provider")
        
        if config.tts_config.speech_rate < 0.5 or config.tts_config.speech_rate > 2.0:
            errors.append("TTS speech rate must be between 0.5 and 2.0")
        
        # Validate search configuration
        if config.search_config.max_search_results < 1:
            errors.append("Max search results must be at least 1")
        
        if config.search_config.search_timeout < 5:
            errors.append("Search timeout must be at least 5 seconds")
        
        # Validate output configuration
        try:
            Path(config.output_config.base_dir)
        except Exception:
            errors.append(f"Invalid output base directory: {config.output_config.base_dir}")
        
        if errors:
            raise ConfigurationError("Configuration validation failed: " + "; ".join(errors))
    
    def _setup_directories(self, config: SystemConfig):
        """Setup required directories.
        
        Args:
            config: System configuration
        """
        directories = [
            config.output_config.base_dir,
            os.path.join(config.output_config.base_dir, "research_reports"),
            os.path.join(config.output_config.base_dir, "podcast_scripts"),
            os.path.join(config.output_config.base_dir, "final_podcasts"),
            os.path.join(config.output_config.base_dir, "audio_segments"),
            os.path.join(config.output_config.base_dir, "temp"),
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.warning(f"Failed to create directory {directory}: {e}")
    
    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """Update configuration at runtime.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._system_config:
                return False
            
            for key, value in updates.items():
                if hasattr(self._system_config, key):
                    setattr(self._system_config, key, value)
                    logger.info(f"Updated configuration: {key} = {value}")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Re-validate after updates
            self._validate_configuration(self._system_config)
            return True
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration (without sensitive data).
        
        Returns:
            Configuration summary dictionary
        """
        if not self._system_config:
            return {"error": "Configuration not loaded"}
        
        return {
            "llm": {
                "provider": self._system_config.llm_config.provider,
                "model": self._system_config.llm_config.model,
                "temperature": self._system_config.llm_config.temperature,
                "max_tokens": self._system_config.llm_config.max_tokens,
                "has_api_key": bool(self._system_config.llm_config.api_key)
            },
            "tts": {
                "provider": self._system_config.tts_config.provider,
                "audio_format": self._system_config.tts_config.audio_format,
                "speech_rate": self._system_config.tts_config.speech_rate,
                "voice_models": list(self._system_config.tts_config.voice_models.keys()),
                "has_api_key": bool(self._system_config.tts_config.api_key)
            },
            "search": {
                "max_results": self._system_config.search_config.max_search_results,
                "max_pages": self._system_config.search_config.max_crawl_pages,
                "timeout": self._system_config.search_config.search_timeout
            },
            "output": {
                "base_dir": self._system_config.output_config.base_dir,
                "formats": {
                    "report": self._system_config.output_config.report_format,
                    "audio": self._system_config.output_config.audio_format
                },
                "cleanup_temp": self._system_config.output_config.cleanup_temp
            }
        }


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_system_config() -> SystemConfig:
    """Get the global system configuration.
    
    Returns:
        System configuration instance
    """
    return config_manager.load_configuration()


def get_environment_config() -> EnvironmentConfig:
    """Get the environment configuration.
    
    Returns:
        Environment configuration instance
    """
    return config_manager.get_environment_config()


def reload_configuration() -> SystemConfig:
    """Reload configuration from environment.
    
    Returns:
        Reloaded system configuration
    """
    config_manager._loaded = False
    config_manager._system_config = None
    return config_manager.load_configuration()


def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present.
    
    Returns:
        Dictionary of API key validation results
    """
    config = get_system_config()
    
    return {
        "llm_api_key": bool(config.llm_config.api_key),
        "tts_api_key": bool(config.tts_config.api_key) if config.tts_config.provider == "qwen" else True,
    }


def setup_logging():
    """Setup logging based on environment configuration."""
    env_config = get_environment_config()
    
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=env_config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file logging if specified
    if env_config.log_file:
        logger.add(
            env_config.log_file,
            level=env_config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="7 days",
            compression="zip"
        )
    
    logger.info("Logging setup completed")