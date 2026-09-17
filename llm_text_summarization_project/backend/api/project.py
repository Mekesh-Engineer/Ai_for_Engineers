from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.services.project_analyzer import project_analyzer
from backend.utils.logger import studio_logger

router = APIRouter(prefix="/api/project", tags=["Project"])

class AnalyzeProjectRequest(BaseModel):
    selected_files: Optional[List[str]] = Field(default=[], description="List of relative file paths to analyze")
    mode: Optional[str] = None

class DebugIssueRequest(BaseModel):
    error_message: str = Field(..., description="Error message or symptom")
    stack_trace: str = Field(..., description="Stack trace or logs")
    relevant_file: Optional[str] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    mode: Optional[str] = None

@router.get("/tree")
async def get_project_tree():
    """Retrieve workspace directory tree structure."""
    return project_analyzer.get_directory_tree()

@router.get("/files")
async def get_project_files():
    """List all workspace files available for context inclusion."""
    return project_analyzer.scan_project_files()

@router.post("/analyze")
async def analyze_project_endpoint(req: AnalyzeProjectRequest):
    """Run full project architectural, dependency, and code smell analysis."""
    try:
        report = await project_analyzer.analyze_project(
            selected_files=req.selected_files,
            mode=req.mode
        )
        return report
    except Exception as e:
        studio_logger.error(f"Project analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Project analysis failed: {str(e)}")

@router.post("/debug")
async def debug_issue_endpoint(req: DebugIssueRequest):
    """Diagnose and resolve code issues using AI debugging assistant."""
    try:
        diagnosis = await project_analyzer.debug_issue(
            error_message=req.error_message,
            stack_trace=req.stack_trace,
            relevant_file=req.relevant_file,
            expected_behavior=req.expected_behavior,
            actual_behavior=req.actual_behavior,
            mode=req.mode
        )
        return diagnosis
    except Exception as e:
        studio_logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug diagnosis failed: {str(e)}")
