"""Wind/cloud-aware scenario-driven model for 1--6 h GHI forecasting."""

from .model import ScenarioDrivenGHIModelV4

ScenarioDrivenGHIModel = ScenarioDrivenGHIModelV4

__all__ = ["ScenarioDrivenGHIModel", "ScenarioDrivenGHIModelV4"]
