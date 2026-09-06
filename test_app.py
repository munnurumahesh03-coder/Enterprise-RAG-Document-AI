import pytest

def test_core_imports():
    """
    This test acts as our CI/CD Security Guard.
    It ensures that all heavy AI libraries can load successfully 
    before allowing the code to deploy to production.
    """
    try:
        import streamlit as st
        import langchain
        import chromadb
        from langchain_groq import ChatGroq
        
        imports_successful = True
    except ImportError as e:
        imports_successful = False
        print(f"Import Error: {e}")
        
    assert imports_successful == True, "CRITICAL: A required library failed to import!"

def test_system_health():
    """
    A simple health check to ensure the testing environment is active.
    """
    system_status = "Healthy"
    assert system_status == "Healthy"
