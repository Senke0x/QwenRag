"""
搜索API路由
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import logging

from api.models.requests import TextSearchRequest, ImageSearchParams
from api.models.responses import TextSearchResponse, ImageSearchResponse, ErrorResponse
from api.services.search_service import SearchService
from api.dependencies import get_search_service
from api.config import api_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

@router.post("/text", response_model=TextSearchResponse)
async def search_by_text(
    request: TextSearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    文本搜索图片
    
    通过自然语言描述搜索相关图片
    
    - **query**: 搜索查询文本
    - **limit**: 返回结果数量 (1-100)
    - **similarity_threshold**: 相似度阈值 (0-1)
    - **filters**: 可选的搜索过滤条件
    """
    try:
        logger.info(f"文本搜索请求: query='{request.query}', limit={request.limit}")
        
        # 验证请求参数
        if request.limit > api_config.max_search_limit:
            raise HTTPException(
                status_code=400,
                detail=f"搜索结果数量超过限制: {request.limit} > {api_config.max_search_limit}"
            )
        
        # 执行搜索
        response = await search_service.search_by_text(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文本搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.post("/image", response_model=ImageSearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="查询图片文件"),
    limit: int = Query(10, description="返回结果数量", ge=1, le=100),
    similarity_threshold: float = Query(0.5, description="相似度阈值", ge=0, le=1),
    include_metadata: bool = Query(True, description="是否包含元数据"),
    search_faces: bool = Query(False, description="是否同时搜索人脸"),
    search_service: SearchService = Depends(get_search_service)
):
    """
    以图搜图
    
    上传图片文件，搜索相似的图片
    
    - **file**: 查询图片文件 (支持 JPEG, PNG, BMP, GIF, WebP)
    - **limit**: 返回结果数量 (1-100)
    - **similarity_threshold**: 相似度阈值 (0-1)
    - **include_metadata**: 是否包含图片元数据
    - **search_faces**: 是否同时搜索图片中的人脸
    """
    try:
        logger.info(f"以图搜图请求: 文件={file.filename}, limit={limit}")
        
        # 验证文件类型
        if file.content_type not in api_config.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 验证文件大小
        file_data = await file.read()
        if len(file_data) > api_config.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {len(file_data)} > {api_config.max_file_size}"
            )
        
        # 验证其他参数
        if limit > api_config.max_search_limit:
            raise HTTPException(
                status_code=400,
                detail=f"搜索结果数量超过限制: {limit} > {api_config.max_search_limit}"
            )
        
        # 创建搜索参数
        params = ImageSearchParams(
            limit=limit,
            similarity_threshold=similarity_threshold,
            include_metadata=include_metadata,
            search_faces=search_faces
        )
        
        # 执行搜索
        response = await search_service.search_by_image(
            image_data=file_data,
            filename=file.filename,
            params=params
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"以图搜图失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/statistics")
async def get_search_statistics(
    search_service: SearchService = Depends(get_search_service)
):
    """
    获取搜索统计信息
    
    返回搜索服务的统计信息和状态
    """
    try:
        logger.info("获取搜索统计信息")
        
        stats = search_service.get_search_statistics()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "搜索统计信息获取成功",
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(f"获取搜索统计信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.get("/health")
async def search_health_check():
    """
    搜索服务健康检查
    
    检查搜索服务的健康状态
    """
    try:
        # 执行基本健康检查
        search_service = get_search_service()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "搜索服务运行正常",
                "service": "SearchService",
                "status": "healthy"
            }
        )
        
    except Exception as e:
        logger.error(f"搜索服务健康检查失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "搜索服务不可用",
                "service": "SearchService", 
                "status": "unhealthy",
                "error": str(e)
            }
        )