"""
人脸识别API路由
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.models.requests import FaceComparisonRequest, FaceSearchRequest
from api.models.responses import (
    FaceDetectionResponse, FaceSearchResponse, FaceComparisonResponse
)
from api.services.face_service import FaceService
from api.dependencies import get_face_service
from api.config import api_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(
    file: UploadFile = File(..., description="图片文件"),
    face_service: FaceService = Depends(get_face_service)
):
    """
    检测图片中的人脸
    
    上传图片文件，检测其中包含的所有人脸
    
    - **file**: 图片文件 (支持 JPEG, PNG, BMP, GIF, WebP)
    
    返回检测到的人脸信息，包括位置坐标和置信度
    """
    try:
        logger.info(f"人脸检测请求: 文件={file.filename}")
        
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
        
        # 执行人脸检测
        response = await face_service.detect_faces(
            image_data=file_data,
            filename=file.filename
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸检测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"人脸检测失败: {str(e)}")

@router.post("/search", response_model=FaceSearchResponse)
async def search_similar_faces(
    file: UploadFile = File(..., description="查询图片文件"),
    limit: int = Query(10, description="返回结果数量", ge=1, le=50),
    similarity_threshold: float = Query(0.8, description="相似度阈值", ge=0, le=1),
    similarity_method: str = Query("cosine_similarity", description="相似度计算方法"),
    face_service: FaceService = Depends(get_face_service)
):
    """
    人脸相似度搜索
    
    上传包含人脸的图片，搜索数据库中相似的人脸
    
    - **file**: 查询图片文件 (支持 JPEG, PNG, BMP, GIF, WebP)
    - **limit**: 返回结果数量 (1-50)
    - **similarity_threshold**: 相似度阈值 (0-1)
    - **similarity_method**: 相似度计算方法
      - cosine_similarity: 余弦相似度 (默认)
      - euclidean_distance: 欧几里得距离  
      - dot_product: 点积相似度
    """
    try:
        logger.info(f"人脸搜索请求: 文件={file.filename}, limit={limit}")
        
        # 验证文件类型和大小
        if file.content_type not in api_config.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        file_data = await file.read()
        if len(file_data) > api_config.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {len(file_data)} > {api_config.max_file_size}"
            )
        
        # 验证参数
        if limit > api_config.max_faces_per_request:
            raise HTTPException(
                status_code=400,
                detail=f"搜索结果数量超过限制: {limit} > {api_config.max_faces_per_request}"
            )
        
        # 创建搜索请求
        search_request = FaceSearchRequest(
            limit=limit,
            similarity_threshold=similarity_threshold,
            similarity_method=similarity_method
        )
        
        # 执行人脸搜索
        response = await face_service.search_similar_faces(
            image_data=file_data,
            filename=file.filename,
            request=search_request
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"人脸搜索失败: {str(e)}")

@router.post("/compare", response_model=FaceComparisonResponse)
async def compare_faces(
    request: FaceComparisonRequest,
    face_service: FaceService = Depends(get_face_service)
):
    """
    人脸相似度比较
    
    比较两个已知人脸ID的相似度
    
    - **face_id_1**: 第一个人脸ID
    - **face_id_2**: 第二个人脸ID  
    - **similarity_method**: 相似度计算方法
      - cosine_similarity: 余弦相似度 (默认)
      - euclidean_distance: 欧几里得距离
      - dot_product: 点积相似度
    
    返回相似度分数和是否判断为同一人
    """
    try:
        logger.info(f"人脸比较请求: {request.face_id_1} vs {request.face_id_2}")
        
        # 执行人脸比较
        response = await face_service.compare_faces(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸比较失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"人脸比较失败: {str(e)}")

@router.get("/similar/{face_id}", response_model=FaceSearchResponse)
async def find_similar_faces_by_id(
    face_id: str,
    limit: int = Query(10, description="返回结果数量", ge=1, le=50),
    similarity_threshold: float = Query(0.8, description="相似度阈值", ge=0, le=1),
    face_service: FaceService = Depends(get_face_service)
):
    """
    根据人脸ID查找相似人脸
    
    通过已知的人脸ID，在数据库中查找相似的人脸
    
    - **face_id**: 查询的人脸ID
    - **limit**: 返回结果数量 (1-50)
    - **similarity_threshold**: 相似度阈值 (0-1)
    """
    try:
        logger.info(f"根据ID查找相似人脸: face_id={face_id}, limit={limit}")
        
        # 验证参数
        if limit > api_config.max_faces_per_request:
            raise HTTPException(
                status_code=400,
                detail=f"搜索结果数量超过限制: {limit} > {api_config.max_faces_per_request}"
            )
        
        # TODO: 实现根据face_id查找相似人脸的功能
        # 这需要在FaceService中添加相应的方法
        
        return FaceSearchResponse(
            success=False,
            message="根据人脸ID查找相似人脸功能正在开发中",
            results=[],
            total_found=0,
            query_time_ms=0,
            query_face_info=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"根据ID查找相似人脸失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查找失败: {str(e)}")

@router.get("/statistics")
async def get_face_statistics(
    face_service: FaceService = Depends(get_face_service)
):
    """
    获取人脸识别统计信息
    
    返回人脸识别服务的统计信息，包括:
    - 数据库中的人脸总数
    - 处理过的图片数量
    - 索引大小等信息
    """
    try:
        logger.info("获取人脸识别统计信息")
        
        stats = face_service.get_face_statistics()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "人脸识别统计信息获取成功",
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(f"获取人脸统计信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.get("/health")
async def face_health_check():
    """
    人脸识别服务健康检查
    
    检查人脸识别服务的健康状态
    """
    try:
        # 执行基本健康检查
        face_service = get_face_service()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "人脸识别服务运行正常",
                "service": "FaceService",
                "status": "healthy"
            }
        )
        
    except Exception as e:
        logger.error(f"人脸识别服务健康检查失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "人脸识别服务不可用",
                "service": "FaceService",
                "status": "unhealthy", 
                "error": str(e)
            }
        )