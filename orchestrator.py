"""
Orchestrator: Sequences all agents in the correct order.

Pipeline:
  Phase 1 - OpenVINO:
    1. DownloadAgent  (once)
    2. For each build type:
       a. BuildAgent
       b. InstallAgent
       c. VerifyAgent (OV tests)

  Phase 2 - ONNX Runtime + OVEP:
    3. OrtDownloadAgent (once)
    4. For each build type:
       a. OrtBuildAgent  (links against OV install)
       b. OrtInstallAgent
       c. VerifyAgent    (OVEP tests)

  5. Generate report
"""

import importlib
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import BuildConfig
from agents import (
    DownloadAgent, BuildAgent, InstallAgent, VerifyAgent,
    OrtDownloadAgent, OrtBuildAgent, OrtInstallAgent,
)
from agents.base_agent import AgentResult
from skills.terminal import TerminalSkill

logger = logging.getLogger(__name__)


class Orchestrator:

    def __init__(self, config: BuildConfig):
        self.config = config
        self.results: Dict[str, List[AgentResult]] = {}
        self.start_time = None

    def run(self, ort_only: bool = False, verify_ov_only: bool = False) -> bool:
        """Run the full pipeline. Returns True if everything passed.
        If ort_only=True, the OpenVINO download/build/install/verify phase is
        skipped and only the ORT+OVEP phase is executed.
        If verify_ov_only=True, only the OV verify step is run for each build type.
        """
        self.start_time = time.time()

        logger.info("=" * 70)
        logger.info("  OpenVINO Build Agent - Orchestrator")
        logger.info(f"  Base directory : {self.config.base_dir}")
        logger.info(f"  Build types    : {self.config.build_types}")
        logger.info(f"  Started        : {datetime.now().isoformat()}")
        logger.info("=" * 70)

        # ── Step 0: Validate Python dependencies in current environment ──
        logger.info("[Orchestrator] Validating Python dependencies...")
        missing = []
        for module_name in ("numpy", "onnx", "packaging"):
            try:
                importlib.import_module(module_name)
            except Exception:
                missing.append(module_name)

        if missing:
            logger.warning(
                "[Orchestrator] Missing modules in active environment: %s. "
                "Install with: python -m pip install %s",
                missing,
                " ".join(missing),
            )

        # ── Step 0b: Configure git proxy ──
        self._configure_git_proxy()

        if verify_ov_only:
            logger.info("[Orchestrator] verify-ov-only mode — running OV verification for each build type.")
            overall_success = True
            for build_type in self.config.build_types:
                logger.info(f"[Orchestrator] >>> Verify OV [{build_type}]")
                verify_agent = VerifyAgent(self.config)
                verify_result = verify_agent.run(build_type=build_type, phase="openvino")
                self.results.setdefault(f"verify_ov_{build_type}", []).append(verify_result)
                if not verify_result.success:
                    overall_success = False
            self._generate_report()
            return overall_success

        if ort_only:
            logger.info("[Orchestrator] ORT-only mode — skipping OpenVINO phase.")
            overall_success = True
        else:
            # ── Step 1: Download (once for all builds) ──
            logger.info("[Orchestrator] >>> STEP 1: Download repository")
            download_agent = DownloadAgent(self.config)
            dl_result = download_agent.run()
            self.results["download"] = [dl_result]

            if not dl_result.success:
                logger.error("[Orchestrator] Download failed. Cannot proceed.")
                self._generate_report()
                return False

        # ── Steps 2-4: Build → Install → Verify (per build type) ──
        overall_success = True

        for build_type in self.config.build_types if not ort_only else []:
            logger.info(f"\n{'='*70}")
            logger.info(f"  Processing: {build_type}")
            logger.info(f"{'='*70}")

            # Build
            logger.info(f"[Orchestrator] >>> STEP 2: Build [{build_type}]")
            build_agent = BuildAgent(self.config)
            build_result = build_agent.run(build_type=build_type)
            self.results.setdefault(f"build_{build_type}", []).append(build_result)

            if not build_result.success:
                logger.error(f"[Orchestrator] Build [{build_type}] failed. Skipping install/verify.")
                overall_success = False
                continue

            # Install
            logger.info(f"[Orchestrator] >>> STEP 3: Install [{build_type}]")
            install_agent = InstallAgent(self.config)
            install_result = install_agent.run(build_type=build_type)
            self.results.setdefault(f"install_{build_type}", []).append(install_result)

            if not install_result.success:
                logger.error(f"[Orchestrator] Install [{build_type}] failed. Skipping verify.")
                overall_success = False
                continue

            # Verify OV
            logger.info(f"[Orchestrator] >>> STEP 4: Verify OV [{build_type}]")
            verify_agent = VerifyAgent(self.config)
            verify_result = verify_agent.run(build_type=build_type, phase="openvino")
            self.results.setdefault(f"verify_ov_{build_type}", []).append(verify_result)

            if not verify_result.success:
                overall_success = False

        # ══════════════════════════════════════════════════════════════
        #  PHASE 2: ONNX Runtime + OpenVINO Execution Provider
        # ══════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("  PHASE 2: ONNX Runtime + OVEP")
        logger.info("=" * 70)

        # Download ORT (once)
        logger.info("[Orchestrator] >>> STEP 5: Download ONNX Runtime")
        ort_dl_agent = OrtDownloadAgent(self.config)
        ort_dl_result = ort_dl_agent.run()
        self.results["ort_download"] = [ort_dl_result]

        if not ort_dl_result.success:
            logger.error("[Orchestrator] ORT download failed. Skipping ORT pipeline.")
            self._generate_report()
            return False

        # Build → Install → Verify ORT per build type and OV device
        for build_type in self.config.ort_build_types:
            # In ort_only mode OV was pre-installed externally — skip the check.
            if not ort_only:
                ov_installed = any(
                    r.success
                    for r in self.results.get(f"install_{build_type}", [])
                )
                if not ov_installed:
                    logger.warning(
                        f"[Orchestrator] Skipping ORT [{build_type}] — "
                        f"OpenVINO install not available"
                    )
                    continue

            for device in self.config.ort_openvino_devices:
                logger.info(f"\n{'='*70}")
                logger.info(f"  ORT-OVEP: {build_type} [{device}]")
                logger.info(f"{'='*70}")

                # Build ORT
                logger.info(f"[Orchestrator] >>> STEP 6: Build ORT-OVEP [{build_type}] [{device}]")
                ort_build_agent = OrtBuildAgent(self.config)
                ort_build_result = ort_build_agent.run(build_type=build_type, device=device)
                self.results.setdefault(f"ort_build_{build_type}_{device}", []).append(ort_build_result)

                if not ort_build_result.success:
                    logger.error(f"[Orchestrator] ORT build [{build_type}] [{device}] failed.")
                    overall_success = False
                    continue

                # Install ORT
                logger.info(f"[Orchestrator] >>> STEP 7: Install ORT-OVEP [{build_type}] [{device}]")
                ort_install_agent = OrtInstallAgent(self.config)
                ort_install_result = ort_install_agent.run(build_type=build_type, device=device)
                self.results.setdefault(f"ort_install_{build_type}_{device}", []).append(ort_install_result)

                if not ort_install_result.success:
                    logger.error(f"[Orchestrator] ORT install [{build_type}] [{device}] failed.")
                    overall_success = False
                    continue

                # Verify OVEP
                logger.info(f"[Orchestrator] >>> STEP 8: Verify OVEP [{build_type}] [{device}]")
                verify_agent = VerifyAgent(self.config)
                verify_result = verify_agent.run(build_type=build_type, phase="ovep", device=device)
                self.results.setdefault(f"verify_ovep_{build_type}_{device}", []).append(verify_result)

                if not verify_result.success:
                    overall_success = False

        # ── Final report ──
        self._generate_report()
        return overall_success

    def _configure_git_proxy(self):
        """Apply corporate proxy settings to the global git config."""
        http_proxy = self.config.http_proxy
        https_proxy = self.config.https_proxy
        if not http_proxy:
            logger.info("[Orchestrator] No proxy configured — skipping git proxy setup.")
            return

        logger.info(f"[Orchestrator] Configuring git proxy: {http_proxy}")
        cmds = [
            f'git config --global http.proxy "{http_proxy}"',
            f'git config --global https.proxy "{https_proxy}"',
        ]
        for cmd in cmds:
            result = TerminalSkill.run(cmd, timeout=30)
            if result.success:
                logger.info(f"[Orchestrator] git config set: {cmd}")
            else:
                logger.warning(
                    f"[Orchestrator] git config command failed: {cmd}\n{result.stderr}"
                )

    def _generate_report(self):
        elapsed = time.time() - self.start_time
        report_path = Path(self.config.base_dir) / "build_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": round(elapsed, 2),
            "total_duration_human": f"{elapsed / 60:.1f} minutes",
            "base_dir": self.config.base_dir,
            "build_types": self.config.build_types,
            "results": {},
        }

        logger.info("\n" + "=" * 70)
        logger.info("  FINAL REPORT")
        logger.info("=" * 70)

        for key, agent_results in self.results.items():
            report["results"][key] = []
            for r in agent_results:
                logger.info(f"  {r}")
                report["results"][key].append({
                    "agent": r.agent_name,
                    "success": r.success,
                    "build_type": r.build_type,
                    "message": r.message,
                    "errors": r.errors,
                })

        logger.info(f"\n  Total time: {elapsed / 60:.1f} minutes")
        logger.info(f"  Report: {report_path}")
        logger.info("=" * 70)

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write report: {e}")
