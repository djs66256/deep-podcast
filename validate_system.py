#!/usr/bin/env python3
"""System validation script for Deep Podcast System."""

import sys
import traceback
from pathlib import Path

def validate_imports():
    """Validate that all core modules can be imported."""
    print("🔍 Validating module imports...")
    
    # Test core models
    try:
        from src.shared.models import (
            ResearchReport, PodcastScript, Character, DialogSegment,
            SystemConfig, TaskStatus
        )
        print("✅ Core data models - OK")
    except Exception as e:
        print(f"❌ Core data models - FAILED: {e}")
        return False
    
    # Test utilities (without external dependencies)
    try:
        from src.shared.utils import (
            generate_task_id, clean_filename, create_topic_hash
        )
        print("✅ Shared utilities - OK")
    except Exception as e:
        print(f"❌ Shared utilities - FAILED: {e}")
        return False
    
    # Test agent states
    try:
        from src.deep_podcast.state import InputState as DPInputState, State as DPState
        from src.deep_research.state import InputState as DRInputState, State as DRState  
        from src.podcast.state import InputState as PInputState, State as PState
        print("✅ Agent state models - OK")
    except Exception as e:
        print(f"❌ Agent state models - FAILED: {e}")
        return False
    
    return True


def validate_structure():
    """Validate project structure."""
    print("\n🏗️  Validating project structure...")
    
    required_dirs = [
        "src/deep_podcast",
        "src/deep_research", 
        "src/podcast",
        "src/shared",
        "tests/unit_tests",
        "tests/integration_tests"
    ]
    
    required_files = [
        "src/deep_podcast/__init__.py",
        "src/deep_podcast/graph.py",
        "src/deep_podcast/state.py",
        "src/deep_research/__init__.py", 
        "src/deep_research/graph.py",
        "src/deep_research/state.py",
        "src/deep_research/tools.py",
        "src/podcast/__init__.py",
        "src/podcast/graph.py", 
        "src/podcast/state.py",
        "src/podcast/tools.py",
        "src/shared/__init__.py",
        "src/shared/models.py",
        "src/shared/utils.py",
        "pyproject.toml",
        "langgraph.json",
        ".env.example"
    ]
    
    base_path = Path(".")
    
    # Check directories
    for dir_path in required_dirs:
        if not (base_path / dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not (base_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure - OK")
    return True


def validate_configuration():
    """Validate configuration files."""
    print("\n⚙️  Validating configuration...")
    
    # Check pyproject.toml
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        if config.get("project", {}).get("name") == "deep-podcast-system":
            print("✅ pyproject.toml - OK")
        else:
            print("❌ pyproject.toml - Invalid project name")
            return False
            
    except Exception as e:
        # Fallback for older Python versions
        with open("pyproject.toml", "r") as f:
            content = f.read()
            if "deep-podcast-system" in content:
                print("✅ pyproject.toml - OK")
            else:
                print("❌ pyproject.toml - Invalid content")
                return False
    
    # Check langgraph.json
    try:
        import json
        with open("langgraph.json", "r") as f:
            config = json.load(f)
        
        expected_graphs = ["deep_podcast", "deep_research", "podcast"]
        if all(graph in config.get("graphs", {}) for graph in expected_graphs):
            print("✅ langgraph.json - OK") 
        else:
            print("❌ langgraph.json - Missing expected graphs")
            return False
            
    except Exception as e:
        print(f"❌ langgraph.json - FAILED: {e}")
        return False
    
    # Check .env.example
    if Path(".env.example").exists():
        print("✅ .env.example - OK")
    else:
        print("❌ .env.example - Missing")
        return False
    
    return True


def validate_data_models():
    """Validate data model functionality."""
    print("\n🗄️  Validating data models...")
    
    try:
        from src.shared.models import ResearchReport, Character, DialogSegment
        from datetime import datetime
        
        # Test ResearchReport
        report = ResearchReport(
            topic="Test Topic",
            summary="Test Summary", 
            key_findings=["Finding 1"],
            sections={"intro": "Introduction"},
            sources=["source1"],
            metadata={"test": True},
            created_at=datetime.now()
        )
        
        # Test serialization
        report_dict = report.to_dict()
        restored_report = ResearchReport.from_dict(report_dict)
        
        assert restored_report.topic == report.topic
        print("✅ ResearchReport model - OK")
        
        # Test Character
        character = Character(
            name="Test Host",
            role="host",
            personality="friendly",
            voice_config={"model": "test"}
        )
        
        char_dict = character.to_dict()
        assert char_dict["name"] == "Test Host"
        print("✅ Character model - OK")
        
        # Test DialogSegment  
        segment = DialogSegment(
            segment_id=1,
            speaker="Test Speaker",
            content="Test content",
            emotion="neutral",
            duration_estimate=10
        )
        
        seg_dict = segment.to_dict()
        assert seg_dict["segment_id"] == 1
        print("✅ DialogSegment model - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Data models validation - FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("🚀 Deep Podcast System Validation")
    print("=" * 40)
    
    validations = [
        validate_structure,
        validate_imports,
        validate_configuration,
        validate_data_models
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed += 1
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Validation Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All validations passed! System is ready.")
        return True
    else:
        print("⚠️  Some validations failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)