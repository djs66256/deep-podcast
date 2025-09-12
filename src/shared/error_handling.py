"""Error handling and recovery mechanisms for the Deep Podcast system.

This module provides comprehensive error handling, retry mechanisms,
circuit breakers, and graceful degradation strategies.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import logging

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category types."""
    NETWORK = "network"
    API = "api"
    PARSING = "parsing"
    VALIDATION = "validation"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime
    component: str
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_message: str
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    retry_count: int = 0


class DeepPodcastException(Exception):
    """Base exception for Deep Podcast system."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        metadata: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.metadata = metadata or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()


class ConfigurationError(DeepPodcastException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            metadata=metadata,
            recoverable=False
        )


class NetworkError(DeepPodcastException):
    """Network-related errors."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            metadata=metadata,
            recoverable=True
        )


class APIError(DeepPodcastException):
    """API service errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        service: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        metadata = metadata or {}
        if status_code:
            metadata["status_code"] = status_code
        if service:
            metadata["service"] = service
            
        severity = ErrorSeverity.HIGH if status_code and status_code >= 500 else ErrorSeverity.MEDIUM
        
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.API,
            metadata=metadata,
            recoverable=True
        )


class ParsingError(DeepPodcastException):
    """Content parsing errors."""
    
    def __init__(self, message: str, content_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        if content_type:
            metadata["content_type"] = content_type
            
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.PARSING,
            metadata=metadata,
            recoverable=True
        )


class ValidationError(DeepPodcastException):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        if field:
            metadata["field"] = field
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.USER_INPUT,
            metadata=metadata,
            recoverable=False
        )


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute synchronous function with circuit breaker protection."""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise DeepPodcastException(
                    "Circuit breaker is OPEN - service unavailable",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.EXTERNAL_SERVICE
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise DeepPodcastException(
                    "Circuit breaker is OPEN - service unavailable",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.EXTERNAL_SERVICE
                )
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context information
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Create error context
        error_context = self._create_error_context(error, component, operation, context)
        
        # Log the error
        self._log_error(error_context)
        
        # Store error history
        self.error_history.append(error_context)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_context)
        
        return recovery_result
    
    def _create_error_context(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Create error context from exception."""
        import traceback
        
        if isinstance(error, DeepPodcastException):
            severity = error.severity
            category = error.category
            metadata = error.metadata
        else:
            severity = self._determine_severity(error)
            category = self._determine_category(error)
            metadata = context or {}
        
        return ErrorContext(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            severity=severity,
            category=category,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            metadata=metadata
        )
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity from exception type."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.LOW
        elif isinstance(error, (PermissionError, FileNotFoundError)):
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _determine_category(self, error: Exception) -> ErrorCategory:
        """Determine error category from exception type."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.SYSTEM
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_message = f"{error_context.component}.{error_context.operation}: {error_context.error_message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log stack trace for high/critical errors
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and error_context.stack_trace:
            logger.debug(f"Stack trace: {error_context.stack_trace}")
    
    def _attempt_recovery(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt to recover from the error."""
        recovery_strategy = self.recovery_strategies.get(error_context.category)
        
        if recovery_strategy:
            try:
                logger.info(f"Attempting recovery for {error_context.category.value} error")
                error_context.recovery_attempted = True
                return recovery_strategy(error_context)
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
        
        return None
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies[ErrorCategory.NETWORK] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.API] = self._recover_api_error
        self.recovery_strategies[ErrorCategory.PARSING] = self._recover_parsing_error
    
    def _recover_network_error(self, error_context: ErrorContext) -> Optional[Any]:
        """Recover from network errors."""
        # Implement exponential backoff retry
        max_retries = 3
        if error_context.retry_count < max_retries:
            wait_time = 2 ** error_context.retry_count
            logger.info(f"Retrying network operation in {wait_time} seconds...")
            # Note: Actual waiting should be handled by the caller
            error_context.retry_count += 1
            return {"retry": True, "wait_time": wait_time}
        
        return None
    
    def _recover_api_error(self, error_context: ErrorContext) -> Optional[Any]:
        """Recover from API errors."""
        metadata = error_context.metadata or {}
        status_code = metadata.get("status_code")
        
        # Retry on 5xx errors
        if status_code and 500 <= status_code < 600:
            if error_context.retry_count < 2:
                wait_time = 5 * (error_context.retry_count + 1)
                logger.info(f"Retrying API call in {wait_time} seconds...")
                # Note: Actual waiting should be handled by the caller
                error_context.retry_count += 1
                return {"retry": True, "wait_time": wait_time}
        
        # Switch to fallback service for 4xx errors
        elif status_code and 400 <= status_code < 500:
            logger.info("Attempting fallback service...")
            return {"fallback": True}
        
        return None
    
    def _recover_parsing_error(self, error_context: ErrorContext) -> Optional[Any]:
        """Recover from parsing errors."""
        # Try alternative parsing method
        metadata = error_context.metadata or {}
        content_type = metadata.get("content_type")
        
        if content_type == "html":
            logger.info("Trying alternative HTML parser...")
            return {"alternative_parser": "text"}
        elif content_type == "json":
            logger.info("Trying fallback JSON parsing...")
            return {"alternative_parser": "manual"}
        
        return None
    
    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker()
        return self.circuit_breakers[service]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_severity": {},
            "by_category": {},
            "by_component": {},
            "recent_errors": []
        }
        
        # Count by severity
        for error in self.error_history:
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            component = error.component
            stats["by_component"][component] = stats["by_component"].get(component, 0) + 1
        
        # Recent errors (last 10)
        recent = sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:10]
        stats["recent_errors"] = [
            {
                "timestamp": error.timestamp.isoformat(),
                "component": error.component,
                "operation": error.operation,
                "severity": error.severity.value,
                "message": error.error_message
            }
            for error in recent
        ]
        
        return stats


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(component: str, operation: str):
    """Decorator for automatic error handling."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, component, operation)
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, component, operation)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def with_circuit_breaker(service: str):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            circuit_breaker = error_handler.get_circuit_breaker(service)
            return await circuit_breaker.call(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            circuit_breaker = error_handler.get_circuit_breaker(service)
            return circuit_breaker.call_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Retry decorators with different strategies
def retry_on_network_error(max_attempts: int = 3):
    """Retry decorator for network errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, NetworkError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )


def retry_on_api_error(max_attempts: int = 3):
    """Retry decorator for API errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type(APIError),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )


def safe_execute(func: Callable, *args, default=None, **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value to return on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe execution failed: {e}")
        return default


async def safe_execute_async(func: Callable, *args, default=None, **kwargs) -> Any:
    """Safely execute an async function with error handling.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        default: Default value to return on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe async execution failed: {e}")
        return default