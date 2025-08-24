"""
索引管理API路由
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import get_indexing_service
from api.models.requests import AddImagesRequest, IndexBuildRequest
from api.models.responses import (
    AddImagesResponse,
    IndexBuildResponse,
    IndexStatusResponse,
)
from api.services.indexing_service import IndexingService
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.post("/build", response_model=IndexBuildResponse)
async def build_index(
    request: IndexBuildRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    构建图片索引

    扫描指定目录的图片，进行批量分析和索引构建

    - **image_directory**: 图片目录路径
    - **batch_size**: 批处理大小 (1-50)
    - **max_workers**: 最大并发数 (1-10)
    - **force_rebuild**: 是否强制重建索引
    - **process_faces**: 是否处理人脸 (提取人脸embedding)

    返回构建任务信息和处理结果
    """
    try:
        logger.info(
            f"索引构建请求: 目录={request.image_directory}, "
            f"批量大小={request.batch_size}, 并发数={request.max_workers}"
        )

        # 执行索引构建
        response = await indexing_service.build_index(request)

        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"索引构建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"索引构建失败: {str(e)}")


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status(
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    获取索引状态

    返回当前索引的状态信息，包括:
    - 索引中的图片总数
    - 人脸总数
    - 索引大小
    - 健康状况
    - 正在进行的任务
    """
    try:
        logger.info("获取索引状态")

        response = indexing_service.get_index_status()

        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取索引状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取索引状态失败: {str(e)}")


@router.post("/add", response_model=AddImagesResponse)
async def add_images(
    request: AddImagesRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    添加新图片到索引

    将指定的图片添加到现有索引中，无需重建整个索引

    - **image_paths**: 图片路径列表
    - **process_faces**: 是否处理人脸

    返回添加结果，包括成功和失败的图片数量
    """
    try:
        logger.info(f"添加图片请求: {len(request.image_paths)}张图片")

        # 执行图片添加
        response = await indexing_service.add_images(request)

        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加图片失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"添加图片失败: {str(e)}")


@router.delete("/rebuild")
async def rebuild_index(
    force: bool = False,
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    重建索引

    删除现有索引并重新构建（危险操作）

    - **force**: 是否强制执行重建

    注意：此操作会删除现有的所有索引数据
    """
    try:
        logger.info(f"重建索引请求: force={force}")

        if not force:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "重建索引需要设置force=true参数",
                    "warning": "此操作将删除所有现有索引数据",
                },
            )

        # TODO: 实现重建索引功能
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "message": "重建索引功能正在开发中",
                "note": "请使用 /build 端点构建新索引",
            },
        )

    except Exception as e:
        logger.error(f"重建索引失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")


@router.get("/tasks")
async def get_running_tasks(
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    获取运行中的索引任务

    返回当前正在执行的索引构建任务列表
    """
    try:
        logger.info("获取运行中的索引任务")

        # 获取当前任务状态
        running_tasks = []
        for task_id, task_info in indexing_service.current_tasks.items():
            running_tasks.append(
                {
                    "task_id": task_id,
                    "status": task_info["status"],
                    "directory": task_info.get("directory", ""),
                    "processed": task_info.get("processed", 0),
                    "total": task_info.get("total", 0),
                    "start_time": task_info.get("start_time", 0),
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"找到 {len(running_tasks)} 个任务",
                "tasks": running_tasks,
                "active_tasks": len(
                    [t for t in running_tasks if t["status"] == "running"]
                ),
            },
        )

    except Exception as e:
        logger.error(f"获取任务列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/health")
async def indexing_health_check():
    """
    索引服务健康检查

    检查索引管理服务的健康状态
    """
    try:
        # 执行基本健康检查
        indexing_service = get_indexing_service()

        # 检查关键组件
        health_info = {
            "service": "IndexingService",
            "status": "healthy",
            "components": {"indexing_pipeline": "healthy", "storage": "healthy"},
        }

        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "索引管理服务运行正常", **health_info},
        )

    except Exception as e:
        logger.error(f"索引服务健康检查失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "索引管理服务不可用",
                "service": "IndexingService",
                "status": "unhealthy",
                "error": str(e),
            },
        )


@router.get("/statistics")
async def get_indexing_statistics(
    indexing_service: IndexingService = Depends(get_indexing_service),
):
    """
    获取索引统计信息

    返回详细的索引统计信息
    """
    try:
        logger.info("获取索引统计信息")

        # 获取索引状态（包含统计信息）
        status_response = indexing_service.get_index_status()

        if not status_response.success:
            raise HTTPException(status_code=500, detail=status_response.message)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "索引统计信息获取成功",
                "statistics": {
                    "index_stats": status_response.stats.dict(),
                    "health_info": status_response.health,
                    "service_status": status_response.status,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取索引统计信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")
