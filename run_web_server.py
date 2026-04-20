#!/usr/bin/env python3
"""
Web Server Launcher for RLPE
启动 RLPE Web 界面的服务器
"""

import sys
from pathlib import Path

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn not installed. Install with: pip install uvicorn fastapi")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from rlpe.api.app import app


def main():
    """Start the RLPE web server."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  🔬 RLPE Web Server                                        ║
    ║  放射虫图版提取系统 - Web 界面                               ║
    ║                                                            ║
    ║  访问地址: http://localhost:8000                            ║
    ║  API 文档: http://localhost:8000/docs                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
