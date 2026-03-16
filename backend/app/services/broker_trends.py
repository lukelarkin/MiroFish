"""
Business Brokerage Trends Prediction Service

Orchestrates the end-to-end prediction pipeline:
    Data Sources → Validation Layer → Seed Content → Ontology → Knowledge Graph → Simulation → Report

The validation layer ensures that predictions are grounded in cross-validated
data from multiple sources with actuarial-style confidence scoring. No single
source can dominate the pipeline. The system refuses to generate predictions
when data quality is insufficient.
"""

import os
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..models.project import ProjectManager, ProjectStatus
from ..models.task import TaskManager, TaskStatus
from .ontology_generator import OntologyGenerator
from .graph_builder import GraphBuilderService
from .simulation_manager import SimulationManager, SimulationStatus
from .simulation_runner import SimulationRunner, RunnerStatus
from .report_agent import ReportAgent, ReportManager, ReportStatus
from .data_pipeline import (
    DataPipeline, LLMSyntheticSource,
    SBADataSource, FREDDataSource, BizBuySellSource, IBBAMarketPulseSource,
)

logger = get_logger('mirofish.broker_trends')


class BrokerTrendService:
    """
    Business Brokerage Trends Prediction Orchestrator

    Collects data from multiple sources, validates and cross-references it,
    then drives validated intelligence through the MiroFish simulation pipeline
    to predict business brokerage market trends.

    Domain: Small/mid-market business acquisitions (not real estate).
    Target user: Business brokers who buy and sell companies.
    """

    SIMULATION_REQUIREMENT = (
        "Simulate how US small and mid-market business brokerage stakeholders -- "
        "business brokers, acquisition entrepreneurs, SBA lenders, business sellers, "
        "private equity searchers, franchise consultants, and M&A advisors -- discuss "
        "and react to current market conditions on social media. Focus on deal flow, "
        "SBA 7(a) lending environment, business valuations and multiples by sector "
        "(HVAC, plumbing, home services, landscaping, auto repair, restaurants, "
        "professional services, manufacturing, e-commerce, SaaS), buyer competition, "
        "seller motivations (retirement, burnout, regulatory pressure), interest rate "
        "impacts on deal financing, and emerging acquisition trends."
    )

    SEED_CONTENT_PROMPT = """You are a business brokerage market analyst specializing in US small
and mid-market business acquisitions.

Write a comprehensive market analysis document (3000-5000 words) about the current US business
brokerage market. This document will be used as source material for a multi-agent social media
simulation, so include specific named entities, organizations, and data points.

Cover all of the following topics:

1. **SBA Lending Environment**: Current SBA 7(a) loan volumes, approval rates, average loan sizes,
   default rates by industry. How the prime rate (currently prime + 2.75% for SBA) affects deal
   financing. Reference SBA district offices and preferred lenders.

2. **Deal Flow & Multiples**: Current asking multiples (SDE, EBITDA, revenue) by sector.
   Reference BizBuySell, BizQuest, DealStats data. Which sectors command premium multiples
   (home services, SaaS) vs discount multiples (restaurants, retail).

3. **Buyer Demand**: Profile of today's business buyer — search fund operators, ETA candidates,
   corporate refugees, private equity platform acquisitions, immigrant entrepreneurs.
   Reference IBBA Market Pulse Survey data on buyer activity.

4. **Seller Motivations**: Baby Boomer retirement wave (Silver Tsunami), COVID burnout sellers,
   regulatory-driven exits, strategic timing. Average age of business sellers, owner dependency
   risk factors.

5. **Hot Sectors for Acquisition**: HVAC, plumbing, electrical, pest control, landscaping,
   auto repair, dental practices, veterinary clinics, home health, IT managed services.
   Why these sectors and what makes them attractive (recurring revenue, essential services,
   fragmented markets, roll-up potential).

6. **Cold Sectors / Risk Areas**: Restaurants, brick-and-mortar retail, print media,
   travel agencies. Why these are challenging and what the risk factors are.

7. **Franchise Resales**: FDD-based valuations, franchise broker networks (Murphy Business,
   Transworld, Sunbelt Business Brokers), territory rights, franchisor approval process.

8. **Due Diligence Red Flags**: Customer concentration, owner dependency, declining revenue trends,
   lease risk, key employee risk, unreported cash/tax issues. How brokers spot and price these.

9. **Market Outlook**: Interest rate trajectory, SBA policy changes, demographic trends affecting
   deal flow over the next 12-24 months. Economic indicators that signal deal volume shifts.

Write in an authoritative, data-rich style. Use specific company names, industry associations
(IBBA, M&A Source, Alliance of M&A Advisors), government agencies (SBA, BLS), and data
providers throughout. Structure the document with clear section headers."""

    # Prediction state storage directory
    PREDICTIONS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'predictions')

    def __init__(self):
        # LLM client is optional — only needed for unvalidated fallback and
        # downstream simulation stages. The FRED data pipeline works without it.
        try:
            self.llm_client = LLMClient()
        except ValueError:
            self.llm_client = None
            logger.warning("LLM client unavailable — only validated data pipeline will work")
        self.task_manager = TaskManager()
        self.pipeline = self._init_pipeline()
        os.makedirs(self.PREDICTIONS_DIR, exist_ok=True)

    def _init_pipeline(self) -> DataPipeline:
        """Initialize the data validation pipeline with all available sources."""
        pipeline = DataPipeline()

        # Register sources in order of trust tier

        # TIER_1: Government / institutional
        pipeline.register_source(SBADataSource(
            api_key=os.environ.get('SBA_API_KEY')
        ))
        pipeline.register_source(FREDDataSource(
            api_key=os.environ.get('FRED_API_KEY')
        ))

        # TIER_2: Industry data providers
        pipeline.register_source(BizBuySellSource(
            api_key=os.environ.get('BIZBUYSELL_API_KEY')
        ))
        pipeline.register_source(IBBAMarketPulseSource())

        # SYNTHETIC: LLM-generated (always last, lowest trust)
        if self.llm_client:
            pipeline.register_source(LLMSyntheticSource(
                llm_client=self.llm_client
            ))

        logger.info(
            f"Data pipeline initialized with {pipeline.registry.source_count} sources. "
            f"Coverage: {pipeline.registry.category_coverage}"
        )
        return pipeline

    def _get_prediction_dir(self, prediction_id: str) -> str:
        pred_dir = os.path.join(self.PREDICTIONS_DIR, prediction_id)
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir

    def _save_prediction_state(self, prediction_id: str, state: Dict[str, Any]):
        pred_dir = self._get_prediction_dir(prediction_id)
        state_file = os.path.join(pred_dir, "state.json")
        state["updated_at"] = datetime.now().isoformat()
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_prediction_state(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        pred_dir = os.path.join(self.PREDICTIONS_DIR, prediction_id)
        state_file = os.path.join(pred_dir, "state.json")
        if not os.path.exists(state_file):
            return None
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_seed_content(self) -> str:
        """
        Generate validated seed content through the data pipeline.

        NEW FLOW:
        1. Run data pipeline (fetch → validate → cross-reference)
        2. If gate is open (enough validated data), use validated seed text
        3. If gate is closed, fall back to LLM generation with a warning
           that the output is UNVALIDATED

        Returns:
            Validated seed text with confidence annotations,
            or unvalidated LLM text with warning header.
        """
        logger.info("Running data pipeline for seed content generation...")

        # Try validated path first
        pipeline_result = self.pipeline.run()

        if pipeline_result["gate_open"]:
            seed_text = pipeline_result["seed_text"]
            stats = pipeline_result["stats"]
            logger.info(
                f"Validated seed content generated: "
                f"{stats.get('actionable', 0)} actionable claims, "
                f"avg confidence {stats.get('avg_confidence', 0):.2f}"
            )
            return seed_text

        # Gate closed — fall back to unvalidated LLM generation
        gate_reason = pipeline_result["gate_reason"]
        logger.warning(
            f"Data pipeline gate CLOSED: {gate_reason}. "
            f"Falling back to unvalidated LLM generation."
        )

        if not self.llm_client:
            # No LLM available either — return what we have from the pipeline
            logger.warning("No LLM client available for fallback. Using raw pipeline data.")
            raw_claims = pipeline_result.get("all_claims", [])
            if raw_claims:
                lines = [f"- {c.get('description', c.get('metric', 'unknown'))}: {c.get('value', 'N/A')}"
                         for c in raw_claims]
                return (
                    "# Data Pipeline Results (Unvalidated)\n\n"
                    f"Gate closed: {gate_reason}\n\n"
                    + "\n".join(lines)
                )
            return (
                "# No Data Available\n\n"
                f"Pipeline gate closed: {gate_reason}\n"
                "No LLM fallback configured. Set FRED_API_KEY in backend/.env "
                "and/or LLM_API_KEY for full functionality."
            )

        messages = [
            {"role": "user", "content": self.SEED_CONTENT_PROMPT}
        ]
        llm_text = self.llm_client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=8192
        )

        # Prepend warning header so downstream consumers know this is unvalidated
        warning_header = (
            "# WARNING: UNVALIDATED CONTENT\n\n"
            f"**Data pipeline gate was CLOSED:** {gate_reason}\n\n"
            "This content was generated by LLM without cross-validation against "
            "real data sources. Confidence intervals are not available. "
            "All claims should be treated as SYNTHETIC with LOW confidence.\n\n"
            "---\n\n"
        )
        return warning_header + llm_text

    def start_prediction(self) -> Dict[str, Any]:
        """
        Create a prediction and launch the pipeline in a background thread.

        Returns:
            dict with prediction_id, task_id, project_id, status
        """
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # Create project
        project = ProjectManager.create_project(name="Business Brokerage Trends Prediction")

        # Create tracking task
        task_id = self.task_manager.create_task(
            task_type="broker_trends_prediction",
            metadata={"prediction_id": prediction_id, "project_id": project.project_id}
        )

        # Initialize prediction state
        pred_state = {
            "prediction_id": prediction_id,
            "project_id": project.project_id,
            "task_id": task_id,
            "status": "started",
            "current_stage": "initializing",
            "simulation_id": None,
            "report_id": None,
            "created_at": datetime.now().isoformat(),
        }
        self._save_prediction_state(prediction_id, pred_state)

        # Launch background thread
        thread = threading.Thread(
            target=self._run_prediction,
            args=(prediction_id, project.project_id, task_id),
            daemon=True
        )
        thread.start()

        return {
            "prediction_id": prediction_id,
            "task_id": task_id,
            "project_id": project.project_id,
            "status": "started"
        }

    def _run_prediction(self, prediction_id: str, project_id: str, task_id: str):
        """Main orchestration method, runs in background thread."""
        pred_state = self._load_prediction_state(prediction_id)

        try:
            # ===== Stage 0: Data validation pipeline (0-5%) =====
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=0,
                message="Running data validation pipeline..."
            )
            pred_state["current_stage"] = "data_validation"
            self._save_prediction_state(prediction_id, pred_state)

            pipeline_result = self.pipeline.run()
            pipeline_health = pipeline_result.get("health", {})

            # Save pipeline health to prediction state for diagnostics
            pred_state["pipeline_health"] = {
                "gate_open": pipeline_result["gate_open"],
                "gate_reason": pipeline_result["gate_reason"],
                "stats": pipeline_result.get("stats", {}),
            }
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=5,
                message=(
                    f"Data pipeline: {'VALIDATED' if pipeline_result['gate_open'] else 'UNVALIDATED (fallback)'} "
                    f"— {pipeline_result['gate_reason']}"
                )
            )

            # ===== Stage 1: Generate seed content (5-10%) =====
            self.task_manager.update_task(
                task_id, progress=6,
                message="Generating business brokerage market analysis..."
            )
            pred_state["current_stage"] = "generating_seed_content"
            self._save_prediction_state(prediction_id, pred_state)

            if pipeline_result["gate_open"]:
                seed_text = pipeline_result["seed_text"]
            else:
                # Fallback to LLM generation
                seed_text = self.generate_seed_content()

            # Save seed text to project
            ProjectManager.save_extracted_text(project_id, seed_text)
            project = ProjectManager.get_project(project_id)
            project.total_text_length = len(seed_text)
            project.simulation_requirement = self.SIMULATION_REQUIREMENT
            ProjectManager.save_project(project)

            self.task_manager.update_task(
                task_id, progress=10,
                message=f"Seed content generated ({len(seed_text)} chars)"
            )

            # ===== Stage 2: Generate ontology (10-20%) =====
            pred_state["current_stage"] = "generating_ontology"
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=12,
                message="Generating ontology from seed content..."
            )

            ontology_gen = OntologyGenerator(llm_client=self.llm_client)
            ontology = ontology_gen.generate(
                document_texts=[seed_text],
                simulation_requirement=self.SIMULATION_REQUIREMENT
            )

            # Save ontology to project
            project = ProjectManager.get_project(project_id)
            project.ontology = ontology
            project.analysis_summary = ontology.get("analysis_summary", "")
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            ProjectManager.save_project(project)

            self.task_manager.update_task(
                task_id, progress=20,
                message=f"Ontology generated: {len(ontology.get('entity_types', []))} entity types"
            )

            # ===== Stage 3: Build knowledge graph (20-55%) =====
            pred_state["current_stage"] = "building_graph"
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=22,
                message="Building knowledge graph..."
            )

            project.status = ProjectStatus.GRAPH_BUILDING
            ProjectManager.save_project(project)

            graph_builder = GraphBuilderService()
            graph_task_id = graph_builder.build_graph_async(
                text=seed_text,
                ontology=ontology,
                graph_name="Business Brokerage Trends Graph"
            )

            # Poll for graph build completion
            while True:
                graph_task = self.task_manager.get_task(graph_task_id)
                if not graph_task:
                    raise RuntimeError("Graph build task disappeared")

                if graph_task.status == TaskStatus.COMPLETED:
                    graph_id = graph_task.result.get("graph_id")
                    break
                elif graph_task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Graph build failed: {graph_task.error}")

                # Map graph progress (0-100) to our range (22-55)
                mapped_progress = 22 + int(graph_task.progress * 0.33)
                self.task_manager.update_task(
                    task_id, progress=mapped_progress,
                    message=f"Building graph: {graph_task.message}"
                )
                time.sleep(5)

            # Update project with graph info
            project = ProjectManager.get_project(project_id)
            project.graph_id = graph_id
            project.graph_build_task_id = graph_task_id
            project.status = ProjectStatus.GRAPH_COMPLETED
            ProjectManager.save_project(project)

            self.task_manager.update_task(
                task_id, progress=55,
                message=f"Knowledge graph built: {graph_id}"
            )

            # ===== Stage 4: Create and prepare simulation (55-70%) =====
            pred_state["current_stage"] = "preparing_simulation"
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=56,
                message="Creating simulation..."
            )

            sim_manager = SimulationManager()
            sim_state = sim_manager.create_simulation(
                project_id=project_id,
                graph_id=graph_id,
                enable_twitter=True,
                enable_reddit=True
            )
            simulation_id = sim_state.simulation_id
            pred_state["simulation_id"] = simulation_id
            self._save_prediction_state(prediction_id, pred_state)

            def sim_prepare_progress(stage, progress, message, **kwargs):
                # Map prepare progress to our range (56-70)
                mapped = 56 + int(progress * 0.14 / 100) if progress <= 100 else 70
                self.task_manager.update_task(
                    task_id, progress=min(mapped, 70),
                    message=f"Preparing simulation [{stage}]: {message}"
                )

            sim_state = sim_manager.prepare_simulation(
                simulation_id=simulation_id,
                simulation_requirement=self.SIMULATION_REQUIREMENT,
                document_text=seed_text,
                use_llm_for_profiles=True,
                progress_callback=sim_prepare_progress
            )

            self.task_manager.update_task(
                task_id, progress=70,
                message=f"Simulation prepared: {sim_state.profiles_count} agent profiles"
            )

            # ===== Stage 5: Run simulation (70-85%) =====
            pred_state["current_stage"] = "running_simulation"
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=71,
                message="Starting simulation..."
            )

            run_state = SimulationRunner.start_simulation(
                simulation_id=simulation_id,
                platform="parallel",
                max_rounds=50  # Reasonable limit for prediction
            )

            # Poll for simulation completion
            while True:
                run_state = SimulationRunner.get_run_state(simulation_id)
                if not run_state:
                    raise RuntimeError("Simulation run state disappeared")

                if run_state.runner_status in [RunnerStatus.COMPLETED, RunnerStatus.STOPPED]:
                    break
                elif run_state.runner_status == RunnerStatus.FAILED:
                    raise RuntimeError(f"Simulation failed: {run_state.error}")

                # Map simulation progress to our range (71-85)
                if run_state.total_rounds > 0:
                    sim_progress = run_state.current_round / run_state.total_rounds
                else:
                    sim_progress = 0
                mapped = 71 + int(sim_progress * 14)
                self.task_manager.update_task(
                    task_id, progress=min(mapped, 85),
                    message=f"Simulation running: round {run_state.current_round}/{run_state.total_rounds}"
                )
                time.sleep(10)

            # Update simulation manager state
            sim_state_updated = sim_manager.get_simulation(simulation_id)
            if sim_state_updated:
                sim_state_updated.status = SimulationStatus.COMPLETED
                sim_manager._save_simulation_state(sim_state_updated)

            self.task_manager.update_task(
                task_id, progress=85,
                message="Simulation completed"
            )

            # ===== Stage 6: Generate report (85-100%) =====
            pred_state["current_stage"] = "generating_report"
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.update_task(
                task_id, progress=86,
                message="Generating analysis report..."
            )

            report_id = f"report_{uuid.uuid4().hex[:12]}"
            pred_state["report_id"] = report_id
            self._save_prediction_state(prediction_id, pred_state)

            agent = ReportAgent(
                graph_id=graph_id,
                simulation_id=simulation_id,
                simulation_requirement=self.SIMULATION_REQUIREMENT
            )

            def report_progress(stage, progress, message):
                mapped = 86 + int(progress * 0.14 / 100) if progress <= 100 else 100
                self.task_manager.update_task(
                    task_id, progress=min(mapped, 99),
                    message=f"Report [{stage}]: {message}"
                )

            report = agent.generate_report(
                progress_callback=report_progress,
                report_id=report_id
            )
            ReportManager.save_report(report)

            if report.status != ReportStatus.COMPLETED:
                raise RuntimeError(f"Report generation failed: {report.error}")

            # ===== Complete =====
            pred_state["status"] = "completed"
            pred_state["current_stage"] = "completed"
            pred_state["report_id"] = report.report_id
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.complete_task(task_id, {
                "prediction_id": prediction_id,
                "project_id": project_id,
                "simulation_id": simulation_id,
                "report_id": report.report_id,
                "graph_id": graph_id,
                "status": "completed",
                "validated": pipeline_result["gate_open"],
            })

            logger.info(f"Broker trends prediction completed: {prediction_id}")

        except Exception as e:
            import traceback
            error_msg = str(e)
            logger.error(f"Broker trends prediction failed: {prediction_id}: {error_msg}")
            logger.error(traceback.format_exc())

            pred_state["status"] = "failed"
            pred_state["error"] = error_msg
            self._save_prediction_state(prediction_id, pred_state)

            self.task_manager.fail_task(task_id, error_msg)

    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get current data pipeline health status."""
        return self.pipeline.run_health_check()
