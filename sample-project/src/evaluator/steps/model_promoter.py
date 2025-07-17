# from zenml import get_step_context, step
# from zenml.client import Client
# from zenml.logger import get_logger

# logger = get_logger(__name__)


# @step
# def promote_model(score: float, stage: str = "production") -> bool:
#     """Model promoter step."""
#     is_promoted = False

#     logger.info(f"Model promoted to {stage}!")
#     is_promoted = True

#     # Get the model in the current context
#     current_model = get_step_context().model

#     # Get the model that is in the production stage
#     client = Client()
#     try:
#         stage_model = client.get_model_version(current_model.name, stage)
#         # compare metrics
#         prod_score = stage_model.get_artifact("lgbm_regressor").run_metadata[
#             "test_accuracy"
#         ]
#         if float(score) > float(prod_score):
#             # If current model has better metrics, promote it
#             is_promoted = True
#             current_model.set_stage(stage, force=True)
#     except KeyError:
#         # If no such model exists, current one is promoted
#         is_promoted = True
#         current_model.set_stage(stage, force=True)
#     return is_promoted
