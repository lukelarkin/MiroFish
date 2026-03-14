"""
Orlando Broker Trends Prediction Service
Orchestrates the end-to-end prediction pipeline:
LLM seed content -> Ontology -> Knowledge Graph -> Simulation -> Report
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

logger = get_logger('mirofish.broker_trends')


class BrokerTrendService:
    """
    Orlando Broker Trends Prediction Orchestrator

    Generates seed content about Orlando real estate via LLM,
    then drives it through the existing MiroFish pipeline.
    """

    SIMULATION_REQUIREMENT = (
        "Simulate how Orlando, Florida real estate market stakeholders -- brokers, "
        "buyers, sellers, investors, developers, and city planners -- discuss and react "
        "to current market trends on social media. Focus on housing prices, inventory, "
        "interest rates, insurance costs, new construction, population growth, "
        "tourism/theme park effects, infrastructure developments, and rental market dynamics."
    )

    SEED_CONTENT_PROMPT = """You are a real estate market research analyst specializing in Central Florida.
Write a comprehensive market analysis document (3000-5000 words) about the current Orlando, Florida
real estate market. This document will be used as source material for a multi-agent social media
simulation, so include specific named entities, organizations, and locations.

Cover all of the following topics with specific data points and named entities:

1. **Housing Market Trends**: Current median home prices, inventory levels, days on market,
   year-over-year changes in Orange, Seminole, Osceola, and Lake counties.

2. **Population Growth & Migration**: In-migration patterns, top origin states, demographic shifts,
   impact on housing demand. Reference Orlando Regional Realtor Association data.

3. **Interest Rate Impacts**: How current Federal Reserve policy affects Central Florida mortgage
   rates, buyer affordability, and market velocity.

4. **Commercial Real Estate**: Office, retail, and industrial sectors. Tech corridor growth along
   I-4, tourism-driven hospitality real estate near International Drive and theme parks.

5. **Rental Market Dynamics**: Average rents, vacancy rates, build-to-rent communities,
   short-term rental regulation (Airbnb/VRBO).

6. **New Construction & Development**: Major builders (DR Horton, Lennar, Pulte, Toll Brothers),
   master-planned communities in Lake Nona, Horizon West, Winter Garden, Kissimmee corridor.
   Mention specific developments and unit counts.

7. **Insurance & Climate Risk**: Homeowners insurance crisis, Citizens Property Insurance,
   hurricane risk premiums, flood zone impacts, roof age requirements.

8. **Tourism & Theme Park Effects**: Walt Disney World, Universal Orlando Resort (Epic Universe),
   Central Florida Tourism Oversight District, impact on local employment and housing demand.

9. **Infrastructure Developments**: SunRail expansion, I-4 Ultimate completion, Orlando
   International Airport South Terminal (Terminal C), Brightline high-speed rail, their effects
   on property values in surrounding neighborhoods.

Write in an authoritative, data-rich style. Use specific neighborhood names, company names,
government agencies, and real organizations throughout. Structure the document with clear section
headers."""

    # Prediction state storage directory
    PREDICTIONS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'predictions')

    def __init__(self):
        self.llm_client = LLMClient()
        self.task_manager = TaskManager()
        os.makedirs(self.PREDICTIONS_DIR, exist_ok=True)

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
        """Generate Orlando real estate market analysis via LLM."""
        messages = [
            {"role": "user", "content": self.SEED_CONTENT_PROMPT}
        ]
        return self.llm_client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=8192
        )

    def start_prediction(self) -> Dict[str, Any]:
        """
        Create a prediction and launch the pipeline in a background thread.

        Returns:
            dict with prediction_id, task_id, project_id, status
        """
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # Create project
        project = ProjectManager.create_project(name="Orlando Broker Trends Prediction")

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
            # ===== Stage 1: Generate seed content (0-10%) =====
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=0,
                message="Generating Orlando real estate market analysis..."
            )
            pred_state["current_stage"] = "generating_seed_content"
            self._save_prediction_state(prediction_id, pred_state)

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
                graph_name="Orlando Broker Trends Graph"
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
                "status": "completed"
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
