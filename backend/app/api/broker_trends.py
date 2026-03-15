"""
Broker Trends API routes
Provides endpoints for business brokerage trends prediction
"""

import traceback
from flask import request, jsonify

from . import broker_trends_bp
from ..services.broker_trends import BrokerTrendService
from ..services.report_agent import ReportManager
from ..models.task import TaskManager
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.broker_trends')


@broker_trends_bp.route('/predict', methods=['POST'])
def predict():
    """
    Start a business brokerage trends prediction.

    Request (JSON):
        {
            "force_regenerate": false   // optional
        }

    Returns:
        {
            "success": true,
            "data": {
                "prediction_id": "pred_xxxx",
                "task_id": "task_xxxx",
                "project_id": "proj_xxxx",
                "status": "started"
            }
        }
    """
    try:
        service = BrokerTrendService()
        result = service.start_prediction()

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.error(f"Failed to start prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@broker_trends_bp.route('/status', methods=['POST'])
def prediction_status():
    """
    Get prediction status.

    Request (JSON):
        {
            "task_id": "task_xxxx",           // option 1
            "prediction_id": "pred_xxxx"      // option 2
        }

    Returns:
        {
            "success": true,
            "data": {
                "status": "processing",
                "progress": 45,
                "message": "...",
                "current_stage": "building_graph"
            }
        }
    """
    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        prediction_id = data.get('prediction_id')

        # If prediction_id provided, load its task_id
        if prediction_id and not task_id:
            service = BrokerTrendService()
            pred_state = service._load_prediction_state(prediction_id)
            if not pred_state:
                return jsonify({
                    "success": False,
                    "error": f"Prediction not found: {prediction_id}"
                }), 404
            task_id = pred_state.get("task_id")

        if not task_id:
            return jsonify({
                "success": False,
                "error": "Please provide task_id or prediction_id"
            }), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify({
                "success": False,
                "error": f"Task not found: {task_id}"
            }), 404

        response_data = task.to_dict()

        # Enrich with prediction state if available
        if prediction_id:
            service = BrokerTrendService()
            pred_state = service._load_prediction_state(prediction_id)
            if pred_state:
                response_data["current_stage"] = pred_state.get("current_stage")
                response_data["prediction_id"] = prediction_id

        return jsonify({
            "success": True,
            "data": response_data
        })

    except Exception as e:
        logger.error(f"Failed to get prediction status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@broker_trends_bp.route('/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id: str):
    """
    Get completed prediction results.

    Returns:
        {
            "success": true,
            "data": {
                "prediction_id": "pred_xxxx",
                "project_id": "proj_xxxx",
                "simulation_id": "sim_xxxx",
                "report_id": "report_xxxx",
                "status": "completed",
                "report_markdown": "..."
            }
        }
    """
    try:
        service = BrokerTrendService()
        pred_state = service._load_prediction_state(prediction_id)

        if not pred_state:
            return jsonify({
                "success": False,
                "error": f"Prediction not found: {prediction_id}"
            }), 404

        response_data = {
            "prediction_id": pred_state["prediction_id"],
            "project_id": pred_state.get("project_id"),
            "simulation_id": pred_state.get("simulation_id"),
            "report_id": pred_state.get("report_id"),
            "status": pred_state.get("status"),
            "current_stage": pred_state.get("current_stage"),
            "created_at": pred_state.get("created_at"),
            "updated_at": pred_state.get("updated_at"),
        }

        # Include report markdown if completed
        report_id = pred_state.get("report_id")
        if report_id and pred_state.get("status") == "completed":
            report = ReportManager.get_report(report_id)
            if report:
                response_data["report_markdown"] = report.markdown_content

        if pred_state.get("error"):
            response_data["error"] = pred_state["error"]

        return jsonify({
            "success": True,
            "data": response_data
        })

    except Exception as e:
        logger.error(f"Failed to get prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@broker_trends_bp.route('/pipeline/health', methods=['GET'])
def pipeline_health():
    """
    Get data pipeline health status.

    Shows all registered data sources, their trust tiers, health metrics,
    circuit breaker states, and overall pipeline viability.

    Returns:
        {
            "success": true,
            "data": {
                "pipeline_viable": true,
                "breaker_states": {...},
                "alerts": [...],
                "source_health": {...}
            }
        }
    """
    try:
        service = BrokerTrendService()
        health = service.get_pipeline_health()
        return jsonify({
            "success": True,
            "data": health
        })
    except Exception as e:
        logger.error(f"Failed to get pipeline health: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
