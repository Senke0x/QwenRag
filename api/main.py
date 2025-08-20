"""
FastAPI应用主入口
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from api.routers import search, faces, indexing
from api.config import APIConfig
from utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    config = APIConfig()
    
    app = FastAPI(
        title="QwenRag API",
        description="智能图像检索系统API - 支持文本搜图、以图搜图和人脸识别",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None
    )
    
    # 添加中间件
    setup_middleware(app, config)
    
    # 注册路由
    register_routers(app)
    
    # 添加全局异常处理
    setup_exception_handlers(app)
    
    # 添加启动和关闭事件
    setup_events(app)
    
    return app

def setup_middleware(app: FastAPI, config: APIConfig):
    """配置中间件"""
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 受信任主机中间件
    if config.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.trusted_hosts
        )
    
    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # 记录请求开始
        logger.info(f"请求开始: {request.method} {request.url}")
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        logger.info(f"请求完成: {request.method} {request.url} - "
                   f"状态码: {response.status_code} - "
                   f"耗时: {process_time:.3f}s")
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

def register_routers(app: FastAPI):
    """注册路由"""
    app.include_router(
        indexing.router,
        prefix="/api/v1/index",
        tags=["索引管理"]
    )
    
    app.include_router(
        search.router,
        prefix="/api/v1/search",
        tags=["搜索"]
    )
    
    app.include_router(
        faces.router,
        prefix="/api/v1/faces",
        tags=["人脸识别"]
    )

def setup_exception_handlers(app: FastAPI):
    """设置全局异常处理器"""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.error(f"参数错误: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": "参数错误", "detail": str(exc)}
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.error(f"文件未找到: {exc}")
        return JSONResponse(
            status_code=404,
            content={"error": "文件未找到", "detail": str(exc)}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"服务器内部错误: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "服务器内部错误", "detail": "请稍后重试"}
        )

def setup_events(app: FastAPI):
    """设置应用事件"""
    
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info("QwenRag API服务启动完成")
        
        # 初始化服务组件
        try:
            # 在这里可以初始化数据库连接、加载模型等
            logger.info("服务组件初始化完成")
        except Exception as e:
            logger.error(f"服务组件初始化失败: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info("QwenRag API服务正在关闭...")
        
        # 清理资源
        try:
            # 在这里可以关闭数据库连接、保存状态等
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

# 创建应用实例
app = create_app()

# 根路径
@app.get("/")
async def root():
    """根路径 - API健康检查"""
    return {
        "message": "QwenRag API服务正在运行",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "search": "/api/v1/search",
            "faces": "/api/v1/faces", 
            "index": "/api/v1/index"
        }
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    # 开发环境直接运行
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )